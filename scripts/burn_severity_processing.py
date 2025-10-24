#!/usr/bin/env python3
"""
Burnt Severity Classification Script (Grouped MultiPolygon Output) — CLI
with per-fire S3 upload and local cleanup.

This script processes Sentinel-2 ARD (Analysis Ready Data) to map the extent
and severity of a bushfire. It compares a pre-fire baseline image with a
post-fire time series to calculate the delta Normalized Burn Ratio (dNBR).

It classifies severity differently for woody vs. grassy landcover types,
masks out bad data (clouds, water), and writes:
  - ONE combined MultiPolygon GeoJSON per original feature (named by fire_name)
  - Optional per-part GeoJSON/COG previews (debug toggles)
  - All outputs are saved into a subfolder: <output_dir>/<fire_name_slug>/
  - Each per-fire subfolder is uploaded to S3 and deleted locally on success
    at: <s3_upload_prefix>/<fire_name_slug>/

Usage examples:
  python burn_severity_cli.py \
    --polygons s3://dea-public-data-dev/projects/burn_cube/derivative/dea_burn_severity/polygon_set/10k_sampled_fires.geojson \
    --upload-to-s3-prefix s3://dea-public-data-dev/projects/burn_cube/derivative/dea_burn_severity/result
"""

# Standard library imports
import os
import sys
import shutil
import argparse
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
import traceback  # For detailed error logging

# Third-party imports
import datacube
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from datacube.utils.cog import write_cog
from datacube.utils.geometry import CRS, Geometry

# Local/custom tool imports
# Assumes 'dea-tools' is in a relative path
sys.path.insert(1, '../../Temp_branch/dea-notebooks/Tools')
try:
    from dea_tools.datahandling import load_ard
    from dea_tools.bandindices import calculate_indices
    from dea_tools.spatial import xr_vectorize
except ImportError as e:
    print("Error: Could not import 'dea-tools'.")
    print("        Please ensure the path added to sys.path is correct.")
    print(f"Details: {e}")
    sys.exit(1)

# =========================
# ======= CONSTANTS =======
# =========================

# Defaults (can be overridden by CLI)
OUTPUT_PRODUCT_DIR = 'products'
MAX_POLYGONS_TO_PROCESS = 10  # Process the first N features
SAVE_PER_PART_GEOJSON = True        # Per-part vector outputs (debug)
SAVE_PER_PART_RASTERS = True        # Per-part COG rasters (debug)
SAVE_COMBINED_PER_FIRE_GEOJSON = True # The new grouped output
FORCE_REBUILD = False               # If True, ignore existing outputs

# S3 upload defaults
DEFAULT_S3_UPLOAD_PREFIX = "s3://dea-public-data-dev/projects/burn_cube/derivative/dea_burn_severity/result"
UPLOAD_TO_S3 = True  # can be disabled via CLI

# Datacube / product parameters
OUTPUT_CRS = 'EPSG:3577'
RESOLUTION = (-10, 10)
S2_PRODUCTS = ['ga_s2am_ard_3', 'ga_s2bm_ard_3', 'ga_s2cm_ard_3']
S2_MEASUREMENTS = [
    'nbart_blue', 'nbart_green', 'nbart_red',
    'nbart_nir_1', 'nbart_nir_2', 'nbart_swir_2', 'nbart_swir_3',
    'oa_nbart_contiguity', 'oa_s2cloudless_mask'
]

# Analysis parameters
PRE_FIRE_BUFFER_DAYS = 50
POST_FIRE_START_DAYS = 15    # Used if no extinguish date
POST_FIRE_WINDOW_DAYS = 60

# Landcover class definitions for "grass"
GRASS_CLASSES = [
    3, 14, 15, 16, 17, 18, 21, 32, 33, 34, 35, 36, 39, 50, 51, 52,
    53, 54, 57, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
    90, 91, 92, 94, 95, 96, 97
]

# Output severity class definitions
# 0 = Unburnt
# 1 = Burnt Grass
# 2 = Low Severity Woody
# 3 = Medium Severity Woody
# 4 = High Severity Woody
# 5 = Very High Severity Woody
# 6 = Masked (Cloud, Water, NoData)


# =============================
# ======= CORE HELPERS ========
# =============================

def _read_geojson_maybe_s3(path: str) -> gpd.GeoDataFrame:
    """
    Read a GeoJSON from a local path or S3 URI into a GeoDataFrame.
    For S3, streams to a temporary local file (requires s3fs).
    """
    if isinstance(path, str) and path.lower().startswith("s3://"):
        try:
            import s3fs  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Reading from s3:// requires the 's3fs' package. "
                "Install with: pip install s3fs"
            ) from e

        gdf = None
        errs = []
        for anon in (True, False):
            try:
                fs = s3fs.S3FileSystem(anon=anon)
                with fs.open(path, "rb") as fsrc, tempfile.NamedTemporaryFile(
                    suffix=".geojson", delete=False
                ) as tmp:
                    shutil.copyfileobj(fsrc, tmp)
                    tmp_path = tmp.name
                gdf = gpd.read_file(tmp_path)
                os.remove(tmp_path)
                break
            except Exception as ee:
                errs.append(str(ee))
                gdf = None
        if gdf is None:
            raise RuntimeError(
                f"Failed to read GeoJSON from S3 path '{path}'. Errors: {errs}"
            )
        return gdf
    else:
        return gpd.read_file(path)


def load_and_prepare_polygons(path: str) -> gpd.GeoDataFrame | None:
    """
    Loads the fire polygon GeoJSON and prepares it for processing.
    Dissolves by 'fire_id' if available to ensure one row per fire.

    Returns:
        GeoDataFrame | None
    """
    print(f"Loading polygons from: {path}")
    try:
        poly_gdf = _read_geojson_maybe_s3(path)
    except FileNotFoundError:
        print(f"Error: Input polygon file not found at {path}")
        return None
    except Exception as e:
        print(f"Error: Failed reading polygons from {path}: {e}")
        return None

    try:
        if len(poly_gdf) > 1 and 'fire_id' in poly_gdf.columns:
            print("Dissolving polygons by 'fire_id'...")
            poly_gdf = poly_gdf.dissolve(by='fire_id', aggfunc='first')
        elif 'fire_id' not in poly_gdf.columns:
            print("Warning: 'fire_id' not in columns. Skipping dissolve.")
    except TypeError as e:
        print(f"Warning: Could not dissolve polygon ({e}). Continuing with loaded data.")

    if 'fire_id' not in poly_gdf.columns:
        poly_gdf["fire_id"] = list(poly_gdf.index)

    return poly_gdf


def load_ard_with_fallback(dc: datacube.Datacube,
                           gpgon: Geometry,
                           time: tuple,
                           min_gooddata_thresholds: list = (0.99, 0.90),
                           **kwargs) -> xr.Dataset:
    """
    Loads ARD data, trying a list of 'min_gooddata' thresholds in order.
    """
    base_params = {
        "dc": dc,
        "products": S2_PRODUCTS,
        "geopolygon": gpgon,
        "time": time,
        "measurements": S2_MEASUREMENTS,
        "output_crs": OUTPUT_CRS,
        "resolution": RESOLUTION,
        "group_by": 'solar_day',
        "cloud_mask": 's2cloudless',
        "dask_chunks": {},
        **kwargs
    }

    data = xr.Dataset()
    for threshold in min_gooddata_thresholds:
        print(f"Attempting load_ard with min_gooddata={threshold} ...")
        base_params['min_gooddata'] = threshold
        data = load_ard(**base_params)
        if getattr(data, "time", xr.DataArray()).size > 0:
            print(f"Success: Loaded {data.time.size} time slices.")
            return data

    print(f"Warning: No data found for time range {time} "
          f"even with min_gooddata={min_gooddata_thresholds[-1]}")
    return data


def calculate_severity(delta_nbr: xr.DataArray,
                       landcover: xr.Dataset,
                       grass_classes: list) -> xr.DataArray:
    """
    Calculates burn severity using different thresholds for woody vs. grass.
    """
    print("Calculating severity based on landcover...")

    # 1) Grass mask from DEA landcover product
    grass_mask = landcover.level4.isin(grass_classes)

    # 2) Woody severity thresholds on dNBR (delta_nbr.NBR)
    sev_woody = xr.zeros_like(delta_nbr.NBR, dtype=np.uint8)
    sev_woody = xr.where(delta_nbr.NBR >= 0.10, 2, sev_woody)  # Low
    sev_woody = xr.where(delta_nbr.NBR >= 0.27, 3, sev_woody)  # Medium
    sev_woody = xr.where(delta_nbr.NBR >= 0.44, 4, sev_woody)  # High
    sev_woody = xr.where(delta_nbr.NBR >= 0.66, 5, sev_woody)  # Very High

    severity_woody_masked = sev_woody.where(grass_mask == 0, 0)

    # 3) Grass severity: binary (class 1)
    severity_grass = grass_mask.where(delta_nbr.NBR >= 0.10, 0)

    # 4) Combine
    severity = severity_woody_masked + severity_grass
    severity.name = 'severity'
    return severity


def create_debug_mask(pre_fire_scene: xr.Dataset,
                      post_fire_stack: xr.Dataset) -> xr.DataArray:
    """
    Creates a mask for pixels to be excluded from the analysis.
    Encodes classes as additive flags (1, 10, 100, 1000, 10000).
    """
    print("Creating debug/masking layer...")

    debug_layer_blank = xr.ones_like(pre_fire_scene.nbart_red, dtype=np.uint16)

    # 1) Water (post-fire): MNDWI > 0
    post_mndwi = calculate_indices(post_fire_stack, index='MNDWI',
                                   collection='ga_s2_3', drop=True)
    max_mndwi = post_mndwi.max('time')
    post_water = debug_layer_blank.where(max_mndwi.MNDWI > 0, 0)  # class 1
    new_debug = post_water

    # 2) Pre-fire cloud
    pre_cloud = (debug_layer_blank.where(
        pre_fire_scene.oa_s2cloudless_mask == 2, 0)) * 10  # class 10
    new_debug = new_debug + pre_cloud

    # 3) Persistent cloud (post-fire)
    post_cloud = post_fire_stack.oa_s2cloudless_mask.where(
        post_fire_stack.oa_s2cloudless_mask >= 1, 1)
    persistent_cloud = post_cloud.min('time')
    post_cloud_mask = (debug_layer_blank.where(persistent_cloud == 2, 0)) * 100
    new_debug = new_debug + post_cloud_mask

    # 4) Pre-fire contiguity
    pre_contiguity = (debug_layer_blank.where(
        pre_fire_scene.oa_nbart_contiguity != 1, 0)) * 1000
    new_debug = new_debug + pre_contiguity

    # 5) Persistent contiguity (post-fire)
    post_contiguity = post_fire_stack.oa_nbart_contiguity.where(
        post_fire_stack.oa_nbart_contiguity == 1, 0)
    persistent_cont = post_contiguity.max('time')
    post_contiguity_mask = (debug_layer_blank.where(persistent_cont != 1, 0)) * 10000
    new_debug = new_debug + post_contiguity_mask

    new_debug.name = 'debug_mask'
    return new_debug


def _append_log(log_path: str, line: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")
        f.flush()            # push to OS buffers
        os.fsync(f.fileno())     # ensure it hits disk


def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not s3_uri.lower().startswith("s3://"):
        raise ValueError(f"Not an S3 URI: {s3_uri}")
    no_scheme = s3_uri[5:]
    bucket, _, key = no_scheme.partition("/")
    return bucket, key.rstrip("/")


def _s3_key_exists_and_nonempty(fs, bucket: str, key: str) -> bool:
    s3_path = f"{bucket}/{key}"
    try:
        if fs.exists(s3_path):
            info = fs.info(s3_path)
            return info.get("Size", 0) > 0
        return False
    except Exception:
        return False


def _upload_dir_to_s3_and_cleanup(local_dir: str, s3_prefix: str) -> bool:
    """
    Upload local_dir recursively to <s3_prefix>/<basename(local_dir)>/*
    and, on success (existence + size checks), delete local_dir.
    """
    if not os.path.isdir(local_dir):
        print(f"[S3 upload] Local directory does not exist: {local_dir}")
        return False
        
    import s3fs  # type: ignore

    bucket, key_prefix = _parse_s3_uri(s3_prefix)          # e.g. ('dea-public-data-dev', '.../result')
    dest_base = key_prefix.strip("/")                       # '.../result'
    slug = os.path.basename(os.path.normpath(local_dir))    # e.g. 'HAMA_Fowlers_East_SFAZ'
    dest_dir_key = f"{dest_base}/{slug}"

    fs = s3fs.S3FileSystem(anon=False)

    # Build manifest of local files (relative paths).
    local_files = []
    for root, _, files in os.walk(local_dir):
        for name in files:
            full = os.path.join(root, name)
            rel = os.path.relpath(full, local_dir).replace("\\", "/")
            size = os.path.getsize(full)
            local_files.append((rel, size, full))

    if not local_files:
        print(f"[S3 upload] Nothing to upload from {local_dir}")
        return False

    # (Optional) create a "folder marker" so some browsers show an actual folder line
    try:
        fs.touch(f"{bucket}/{dest_dir_key}/")
    except Exception:
        pass  # not required

    # Upload each file to the EXACT key we want: <prefix>/<slug>/<rel>
    print(f"[S3 upload] Uploading '{local_dir}' -> 's3://{bucket}/{dest_dir_key}/' ...")
    for rel, _, full in local_files:
        remote_key = f"{dest_dir_key}/{rel}"
        fs.put(full, f"{bucket}/{remote_key}")

    import shutil
    shutil.rmtree(local_dir, ignore_errors=True)
    return True


def process_single_fire(
    fire_series: pd.Series,
    poly_crs: CRS,
    dc: datacube.Datacube,
    unique_fire_name: str,
    save_per_part_vectors: bool = SAVE_PER_PART_GEOJSON,
    save_per_part_rasters: bool = SAVE_PER_PART_RASTERS,
    log_path: str | None = None,
    # per-fire output directory to place per-part files in the subfolder
    out_dir: str | None = None,
) -> gpd.GeoDataFrame | None:
    """
    Full burn mapping workflow for a single polygon (part).
    Returns:
        GeoDataFrame dissolved by 'severity' (with 'severity' column),
        reprojected to 'EPSG:4283', or None if nothing to save.
    """

    # Set AWS credentials for accessing public data
    os.environ["AWS_NO_SIGN_REQUEST"] = "Yes"

    # --- Geometry and metadata
    gpgon = Geometry(fire_series.geometry, crs=poly_crs)

    # Make a single-row GeoDataFrame for clipping; ensure CRS
    poly = gpd.GeoDataFrame([fire_series], crs=poly_crs).copy()
    poly = poly.to_crs('EPSG:4283')  # ensure clip CRS matches vectors

    fire_id = fire_series.get('fire_id', None)
    fire_name_part = unique_fire_name

    # --- Dates
    try:
        fire_date = str(fire_series.ignition_date)[:10]
    except AttributeError:
        print("Error: Could not find 'ignition_date' column in input polygons.")
        raise

    try:
        if pd.isna(fire_series.extinguish_date):
            extinguish_date = 'None'
        else:
            extinguish_date = str(fire_series.extinguish_date)[:10]
    except (AttributeError, KeyError):
        extinguish_date = 'None'

    # --- Time windows
    start_date_pre = (datetime.strptime(fire_date, '%Y-%m-%d')
                      - timedelta(days=PRE_FIRE_BUFFER_DAYS)).strftime('%Y-%m-%d')
    end_date_pre = (datetime.strptime(fire_date, '%Y-%m-%d')
                    - timedelta(days=1)).strftime('%Y-%m-%d')

    if extinguish_date == 'None':
        start_date_post = (datetime.strptime(fire_date, '%Y-%m-%d')
                           + timedelta(days=POST_FIRE_START_DAYS)).strftime('%Y-%m-%d')
    else:
        start_date_post = extinguish_date

    end_date_post = (datetime.strptime(start_date_post, '%Y-%m-%d')
                     + timedelta(days=POST_FIRE_WINDOW_DAYS)).strftime('%Y-%m-%d')

    # Landcover year: use previous year for Jan–Sep ignitions
    month_number = int(fire_date[5:7])
    landcover_year = fire_date[0:4] if month_number >= 10 else str(int(fire_date[0:4]) - 1)

    # --- Load data
    baseline = load_ard_with_fallback(dc, gpgon, time=(start_date_pre, end_date_pre),
                                      min_gooddata_thresholds=(0.99, 0.90))
    if baseline.time.size == 0:
        if log_path:
            _append_log(
                log_path,
                f"{fire_name_part}\tbaseline_scenes=0\tpost_scenes=0\tgrid=0x0\ttotal_px=0\tvalid_px=0\tburn_px=0\tmasked_px=0"
            )
        print("No baseline data for this part. Skipping.")
        return None
    closest_bl = baseline.isel(time=-1)

    post = load_ard_with_fallback(dc, gpgon, time=(start_date_post, end_date_post),
                                  min_gooddata_thresholds=(0.90,))
    if post.time.size == 0:
        if log_path:
            yy = int(closest_bl.sizes.get('y', 0))
            xx = int(closest_bl.sizes.get('x', 0))
            total_px = yy * xx
            bl_valid_px = int((closest_bl.oa_nbart_contiguity == 1).sum().item()) if 'oa_nbart_contiguity' in closest_bl.data_vars else 0
            _append_log(
                log_path,
                f"{fire_name_part}\tbaseline_scenes={baseline.time.size}\tpost_scenes=0\tgrid={yy}x{xx}\ttotal_px={total_px}\tvalid_px_baseline={bl_valid_px}\tburn_px=0\tmasked_px=0"
            )
        print("No post-fire data for this part. Skipping.")
        return None

    landcover = dc.load(
        product='ga_ls_landcover_class_cyear_3',
        geopolygon=gpgon,
        time=(landcover_year),
        output_crs=OUTPUT_CRS,
        resolution=RESOLUTION,
        group_by='solar_day',
        dask_chunks={}
    )
    if landcover.time.size == 0:
        if log_path:
            yy = int(closest_bl.sizes.get('y', 0))
            xx = int(closest_bl.sizes.get('x', 0))
            total_px = yy * xx
            _append_log(
                log_path,
                f"{fire_name_part}\tbaseline_scenes={baseline.time.size}\tpost_scenes={post.time.size}\tgrid={yy}x{xx}\ttotal_px={total_px}\tvalid_px=0\tburn_px=0\tmasked_px=0\tlandcover=missing"
            )
        print(f"No landcover data for year {landcover_year}. Skipping.")
        return None
    landcover = landcover.isel(time=0)

    # --- Indices and severity
    pre_nbr = calculate_indices(closest_bl, index='NBR', collection='ga_s2_3', drop=True)
    post_nbr = calculate_indices(post, index='NBR', collection='ga_s2_3', drop=True)
    min_post_nbr = post_nbr.min('time')
    delta_nbr = pre_nbr - min_post_nbr

    severity = calculate_severity(delta_nbr, landcover, GRASS_CLASSES)

    # --- Masking
    debug_mask = create_debug_mask(closest_bl, post)
    final_severity = severity.where(debug_mask == 0, 6)
    final_severity.name = 'burn_severity'

    # --- Per-part pixel stats (for log)
    try:
        yy = int(final_severity.sizes.get('y', 0))
        xx = int(final_severity.sizes.get('x', 0))
        total_px = yy * xx
    except Exception:
        yy = xx = total_px = 0

    try:
        valid_px = int((debug_mask == 0).sum().item())
    except Exception:
        valid_px = 0

    try:
        burn_px = int(final_severity.isin([1, 2, 3, 4, 5]).sum().item())
    except Exception:
        burn_px = 0

    try:
        masked_px = int((final_severity == 6).sum().item())
    except Exception:
        masked_px = 0

    try:
        bl_valid_px = int((closest_bl.oa_nbart_contiguity == 1).sum().item()) if 'oa_nbart_contiguity' in closest_bl.data_vars else 0
        post_any_contig = post.oa_nbart_contiguity.max('time') if 'oa_nbart_contiguity' in post.data_vars else None
        post_valid_px_any = int((post_any_contig == 1).sum().item()) if post_any_contig is not None else 0
    except Exception:
        bl_valid_px = 0
        post_valid_px_any = 0

    if log_path:
        _append_log(
            log_path,
            (
                f"{fire_name_part}"
                f"\tbaseline_scenes={baseline.time.size}"
                f"\tpost_scenes={post.time.size}"
                f"\tgrid={yy}x{xx}"
                f"\ttotal_px={total_px}"
                f"\tvalid_px={valid_px}"
                f"\tburn_px={burn_px}"
                f"\tmasked_px={masked_px}"
                f"\tvalid_px_baseline={bl_valid_px}"
                f"\tvalid_px_post_any={post_valid_px_any}"
            )
        )

    # --- Vectorize (exclude unburnt class 0)
    print("Vectorizing severity raster (per-part)...")
    severity_vectors = xr_vectorize(
        final_severity,
        attribute_col='severity',
        crs=OUTPUT_CRS,
        mask=final_severity != 0
    )
    if severity_vectors.empty:
        print("No burn area detected for this part.")
        return None

    # Reproject vectors to geographic CRS for output & clip to part geometry
    severity_vectors = severity_vectors.to_crs('EPSG:4283')
    clipped = severity_vectors.clip(poly)

    # Dissolve by severity so each severity class is a (Multi)Polygon
    aggregated = clipped.dissolve(by='severity').reset_index()

    # Add metadata for traceability
    aggregated['fire_id'] = fire_id
    aggregated['fire_name'] = fire_series.get('fire_name', fire_name_part)
    aggregated['ignition_date'] = fire_date
    aggregated['extinguish_date'] = extinguish_date

    # --- Optional per-part saves (under out_dir)
    base_dir = out_dir if out_dir else OUTPUT_PRODUCT_DIR
    os.makedirs(base_dir, exist_ok=True)

    if save_per_part_vectors:
        out_vec = os.path.join(base_dir, f'burn_severity_polygons_{fire_name_part}.geojson')
        aggregated.to_file(out_vec, driver='GeoJSON')
        print(f"Saved per-part severity GeoJSON: {out_vec}")

    if save_per_part_rasters:
        out_cog_preview = os.path.join(base_dir, f's2_postfire_preview_{fire_name_part}.tif')
        write_cog(post.isel(time=0).to_array().compute(), fname=out_cog_preview, overwrite=True)
        print(f"Saved post-fire preview COG: {out_cog_preview}")

        out_cog_debug = os.path.join(base_dir, f'debug_mask_raster_{fire_name_part}.tif')
        write_cog(debug_mask.compute(), fname=out_cog_debug, overwrite=True)
        print(f"Saved debug mask COG: {out_cog_debug}")

    print(f"Successfully processed part: {fire_name_part}")
    return aggregated


def _is_valid_geojson(path: str) -> bool:
    """
    Check if a GeoJSON exists, is non-empty, and has severity column.
    """
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return False
        gdf = gpd.read_file(path)
        return (len(gdf) > 0) and ("severity" in gdf.columns)
    except Exception:
        return False


# =========================
# ========= MAIN ==========
# =========================

def main(polygons_path: str,
         output_dir: str = OUTPUT_PRODUCT_DIR,
         max_fires: int = MAX_POLYGONS_TO_PROCESS,
         save_per_part_vectors: bool = SAVE_PER_PART_GEOJSON,
         save_per_part_rasters: bool = SAVE_PER_PART_RASTERS,
         save_combined: bool = SAVE_COMBINED_PER_FIRE_GEOJSON,
         force_rebuild: bool = FORCE_REBUILD,
         upload_to_s3: bool = UPLOAD_TO_S3,
         s3_upload_prefix: str = DEFAULT_S3_UPLOAD_PREFIX,
         app_name: str = "Burnt_Area_Mapping"):
    # Bind runtime options to globals (kept minimal)
    global OUTPUT_PRODUCT_DIR, MAX_POLYGONS_TO_PROCESS
    global SAVE_PER_PART_GEOJSON, SAVE_PER_PART_RASTERS
    global SAVE_COMBINED_PER_FIRE_GEOJSON, FORCE_REBUILD

    OUTPUT_PRODUCT_DIR = output_dir
    MAX_POLYGONS_TO_PROCESS = max_fires
    SAVE_PER_PART_GEOJSON = save_per_part_vectors
    SAVE_PER_PART_RASTERS = save_per_part_rasters
    SAVE_COMBINED_PER_FIRE_GEOJSON = save_combined
    FORCE_REBUILD = force_rebuild

    os.makedirs(OUTPUT_PRODUCT_DIR, exist_ok=True)
    print(f"All outputs will be saved to: {OUTPUT_PRODUCT_DIR}")

    # If uploading, prep S3 fs and a skip check against S3 for combined file
    s3_fs = None
    if upload_to_s3:
        try:
            import s3fs  # type: ignore
            s3_fs = s3fs.S3FileSystem(anon=False)
        except Exception as e:
            print("Warning: '--upload-to-s3-prefix' requested but 's3fs' not available.")
            print("         Install with: pip install s3fs")
            upload_to_s3 = False

    dc = datacube.Datacube(app=app_name)

    all_polys = load_and_prepare_polygons(polygons_path)
    if all_polys is None or all_polys.empty:
        print("No polygons loaded. Exiting.")
        return

    # Limit number of features for this run
    num_fires_to_process = min(MAX_POLYGONS_TO_PROCESS, len(all_polys))
    print(f"Found {len(all_polys)} total features. Processing first {num_fires_to_process}.")
    polys_to_process = all_polys.iloc[:num_fires_to_process]

    # Explode MultiPolygons into individual parts and keep a MultiIndex
    try:
        all_polys_exploded = polys_to_process.explode(index_parts=True)
    except TypeError:
        print("GeoPandas without index_parts: building a MultiIndex fallback...")
        tmp = polys_to_process.explode()
        tmp["__part_id__"] = tmp.groupby(tmp.index).cumcount()
        all_polys_exploded = tmp.set_index([tmp.index, "__part_id__"])
        all_polys_exploded.index.names = [None, None]

    print(f"Exploded into {len(all_polys_exploded)} polygon parts.")

    # Group parts by original pre-explosion row (level=0 of MultiIndex)
    if isinstance(all_polys_exploded.index, pd.MultiIndex):
        group_iter = all_polys_exploded.groupby(level=0, sort=False)
    else:
        # No MultiIndex: each row is its own group
        group_iter = [(idx, all_polys_exploded.loc[[idx]]) for idx in all_polys_exploded.index]

    part_success = part_fail = 0
    combined_success = combined_skip = 0

    print("\nBeginning grouped processing (combine parts per original feature)...")
    for orig_idx, parts_df in group_iter:
        # Pick stable metadata/name from the original (pre-explosion) row
        orig_row = polys_to_process.loc[orig_idx]

        # Choose a group name (prefer fire_name; fallback to fire_id or row index)
        if 'fire_name' in orig_row and pd.notna(orig_row['fire_name']):
            base_fire_name = str(orig_row['fire_name']).strip()
        elif 'fire_id' in orig_row:
            base_fire_name = f"fire_id_{orig_row['fire_id']}"
        else:
            base_fire_name = f"fire_{orig_idx}"

        # Very simple file slug: collapse whitespace to underscores; remove path separators
        base_fire_slug = "_".join(base_fire_name.split())
        base_fire_slug = base_fire_slug.replace(os.sep, "_")
        if os.altsep:
            base_fire_slug = base_fire_slug.replace(os.altsep, "_")

        # Per-fire subfolder
        group_dir = os.path.join(OUTPUT_PRODUCT_DIR, base_fire_slug)
        os.makedirs(group_dir, exist_ok=True)

        combined_path = os.path.join(group_dir, f"burn_severity_polygons_{base_fire_slug}.geojson")

        # Per-group log file (inside the subfolder)
        log_path = os.path.join(group_dir, f"{base_fire_slug}_processing.log")
        if not os.path.exists(log_path):
            _append_log(log_path, f"# Processing log for {base_fire_name} (group index {orig_idx})")
            _append_log(log_path, "# part_name\tbaseline_scenes\tpost_scenes\tgrid\ttotal_px\tvalid_px\tburn_px\tmasked_px\tvalid_px_baseline\tvalid_px_post_any")

        # SKIP check:
        #  1) local combined valid?
        #  2) or (if uploading) S3 combined exists (non-empty)?
        skip_due_to_output = False
        if (not FORCE_REBUILD) and SAVE_COMBINED_PER_FIRE_GEOJSON:
            if _is_valid_geojson(combined_path):
                print(f"[Group '{base_fire_name}'] Local combined exists & valid. Skipping processing.")
                combined_skip += 1
                skip_due_to_output = True
            elif upload_to_s3 and s3_fs is not None:
                bucket, prefix = _parse_s3_uri(s3_upload_prefix)
                remote_key = f"{prefix}/{base_fire_slug}/burn_severity_polygons_{base_fire_slug}.geojson"
                if _s3_key_exists_and_nonempty(s3_fs, bucket, remote_key):
                    print(f"[Group '{base_fire_name}'] Combined exists in S3. Skipping processing.")
                    combined_skip += 1
                    skip_due_to_output = True

        if skip_due_to_output:
            # If local folder exists but we're skipping due to S3, we can optionally clean it up
            # (keep it to avoid losing logs / artifacts unintentionally)
            continue

        print("\n" + "="*80)
        print(f"Processing original feature group: '{base_fire_name}'")
        print("="*80)

        per_part_gdfs: list[gpd.GeoDataFrame] = []

        # Iterate each polygon part in this original feature
        for (orig_idx2, part_id), fire_series in parts_df.iterrows():
            unique_fire_name = f"{base_fire_slug}_part_{part_id}"
            try:
                gdf_part = process_single_fire(
                    fire_series=fire_series,
                    poly_crs=all_polys.crs,
                    dc=dc,
                    unique_fire_name=unique_fire_name,
                    save_per_part_vectors=SAVE_PER_PART_GEOJSON,
                    save_per_part_rasters=SAVE_PER_PART_RASTERS,
                    log_path=log_path,
                    out_dir=group_dir,  # save per-part outputs inside subfolder
                )
                if gdf_part is not None and len(gdf_part) > 0:
                    per_part_gdfs.append(gdf_part)
                    part_success += 1
            except Exception as e:
                part_fail += 1
                print(f"!!! FAILED to process part '{unique_fire_name}': {e}")
                traceback.print_exc()
                print("Continuing to next part...")

        # Combine parts for this group into one MultiPolygon per severity
        wrote_combined = False
        if SAVE_COMBINED_PER_FIRE_GEOJSON:
            if per_part_gdfs:
                try:
                    crs_out = per_part_gdfs[0].crs or 'EPSG:4283'
                    combined_gdf = gpd.GeoDataFrame(
                        pd.concat(per_part_gdfs, ignore_index=True), crs=crs_out
                    )
                    combined_gdf = combined_gdf.dissolve(by='severity').reset_index()

                    # Attach group-level metadata from original row
                    combined_gdf['fire_id'] = orig_row.get('fire_id', None)
                    combined_gdf['fire_name'] = base_fire_name

                    ign = orig_row.get('ignition_date', None)
                    ext = orig_row.get('extinguish_date', None)
                    combined_gdf['ignition_date'] = (str(ign)[:10] if pd.notna(ign) else "")
                    combined_gdf['extinguish_date'] = ("None" if (ext is None or pd.isna(ext)) else str(ext)[:10])

                    # Atomic write: write to temp then replace (inside subfolder)
                    tmp_path = combined_path + ".tmp"
                    combined_gdf.to_file(tmp_path, driver="GeoJSON")
                    os.replace(tmp_path, combined_path)

                    print(f"[COMBINED] Saved MultiPolygon GeoJSON: {combined_path}")
                    _append_log(log_path, f"# Combined saved: {combined_path}")
                    combined_success += 1
                    wrote_combined = True
                except Exception as e:
                    print(f"!!! FAILED to save combined GeoJSON for '{base_fire_name}': {e}")
                    traceback.print_exc()
            else:
                print(f"No per-part severity vectors to combine for '{base_fire_name}'.")
                _append_log(log_path, "# No per-part vectors to combine.")

        # Upload this group's folder to S3 and remove local copy if successful
        if upload_to_s3:
            ok = _upload_dir_to_s3_and_cleanup(group_dir, s3_upload_prefix)
            if not ok:
                print(f"[S3 upload] WARNING: '{group_dir}' was not removed (upload verification failed).")

    print("\n" + "="*80)
    print("Batch processing complete.")
    print(f"  Per-part success: {part_success}")
    print(f"  Per-part failed:  {part_fail}")
    if SAVE_COMBINED_PER_FIRE_GEOJSON:
        print(f"  Combined (group) success: {combined_success}")
        print(f"  Combined (group) skipped (exists): {combined_skip}")
    print("="*88)


# =========================
# ======== CLI ============
# =========================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Burn severity mapping over Sentinel-2 ARD with grouped MultiPolygon outputs, per-fire S3 upload, and local cleanup."
    )
    parser.add_argument(
        "--polygons", required=True,
        help="Path to input polygons GeoJSON. Local path or S3 URI (s3://...)."
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_PRODUCT_DIR,
        help=f"Output base directory (default: {OUTPUT_PRODUCT_DIR})."
    )
    parser.add_argument(
        "--max-fires", type=int, default=MAX_POLYGONS_TO_PROCESS,
        help=f"Process at most this many features (default: {MAX_POLYGONS_TO_PROCESS})."
    )
    bool_action = argparse.BooleanOptionalAction if hasattr(argparse, "BooleanOptionalAction") else None
    parser.add_argument(
        "--save-per-part-vectors",
        action=bool_action or "store_true",
        default=SAVE_PER_PART_GEOJSON,
        help=f"Save per-part GeoJSONs (default: {SAVE_PER_PART_GEOJSON}). "
             f"Use --no-save-per-part-vectors to disable if supported."
    )
    parser.add_argument(
        "--save-per-part-rasters",
        action=bool_action or "store_true",
        default=SAVE_PER_PART_RASTERS,
        help=f"Save per-part COG rasters (default: {SAVE_PER_PART_RASTERS}). "
             f"Use --no-save-per-part-rasters to disable if supported."
    )
    parser.add_argument(
        "--save-combined-per-fire",
        action=bool_action or "store_true",
        default=SAVE_COMBINED_PER_FIRE_GEOJSON,
        help=f"Save combined MultiPolygon GeoJSON per fire (default: {SAVE_COMBINED_PER_FIRE_GEOJSON}). "
             f"Use --no-save-combined-per-fire to disable if supported."
    )
    parser.add_argument(
        "--force-rebuild",
        action=bool_action or "store_true",
        default=FORCE_REBUILD,
        help=f"Rebuild even if outputs exist (default: {FORCE_REBUILD}). "
             f"Use --no-force-rebuild to skip rebuilding if supported."
    )
    parser.add_argument(
        "--upload-to-s3-prefix",
        default=DEFAULT_S3_UPLOAD_PREFIX,
        help=f"S3 prefix to upload each per-fire subfolder to (default: {DEFAULT_S3_UPLOAD_PREFIX})."
    )
    parser.add_argument(
        "--upload-to-s3",
        action=bool_action or "store_true",
        default=UPLOAD_TO_S3,
        help=f"Enable S3 upload and local cleanup (default: {UPLOAD_TO_S3}). "
             f"Use --no-upload-to-s3 to disable if supported."
    )
    parser.add_argument(
        "--app-name", default="Burnt_Area_Mapping",
        help="Datacube app name (default: Burnt_Area_Mapping)."
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(
        polygons_path=args.polygons,
        output_dir=args.output_dir,
        max_fires=args.max_fires,
        save_per_part_vectors=args.save_per_part_vectors,
        save_per_part_rasters=args.save_per_part_rasters,
        save_combined=args.save_combined_per_fire,
        force_rebuild=args.force_rebuild,
        upload_to_s3=args.upload_to_s3,
        s3_upload_prefix=args.upload_to_s3_prefix,
        app_name=args.app_name,
    )