"""
Burnt Severity Classification Script (Grouped MultiPolygon Output)

This script processes Sentinel-2 ARD (Analysis Ready Data) to map the extent
and severity of a bushfire. It compares a pre-fire baseline image with a
post-fire time series to calculate the delta Normalized Burn Ratio (dNBR).

It classifies severity differently for woody vs. grassy landcover types,
masks out bad data (clouds, water), and writes:
  - ONE combined MultiPolygon GeoJSON per original feature (named by fire_name)
  - Optional per-part GeoJSON/COG previews (off by default)

Key implementation detail:
- Uses GeoPandas `explode(index_parts=True)` so each MultiPolygon becomes
  multiple polygon rows with a MultiIndex (original_row_id, part_id).
- Groups parts back with `groupby(level=0)` and dissolves by `severity`
  to produce MultiPolygon geometries per severity class.
"""

# Standard library imports
import os
import sys
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
    print("       Please ensure the path added to sys.path is correct.")
    print(f"Details: {e}")
    sys.exit(1)

# =========================
# ======= CONSTANTS =======
# =========================

# Input data
POLYGON_PATH = '10k_sampled_fires.geojson'

# Batch processing setting
MAX_POLYGONS_TO_PROCESS = 10  # Process the first N features

# Output directory
OUTPUT_PRODUCT_DIR = 'products'

# Output toggles
SAVE_PER_PART_GEOJSON = True         # Per-part vector outputs (debug)
SAVE_PER_PART_RASTERS = True         # Per-part COG rasters (debug)
SAVE_COMBINED_PER_FIRE_GEOJSON = True # The new grouped output

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
POST_FIRE_START_DAYS = 15   # Used if no extinguish date
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

def load_and_prepare_polygons(path: str) -> gpd.GeoDataFrame | None:
    """
    Loads the fire polygon GeoJSON and prepares it for processing.
    Dissolves by 'fire_id' if available to ensure one row per fire.

    Returns:
        GeoDataFrame | None
    """
    print(f"Loading polygons from: {path}")
    try:
        poly_gdf = gpd.read_file(path)
    except FileNotFoundError:
        print(f"Error: Input polygon file not found at {path}")
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


def process_single_fire(
    fire_series: pd.Series,
    poly_crs: CRS,
    dc: datacube.Datacube,
    unique_fire_name: str,
    save_per_part_vectors: bool = SAVE_PER_PART_GEOJSON,
    save_per_part_rasters: bool = SAVE_PER_PART_RASTERS
) -> gpd.GeoDataFrame | None:
    """
    Full burn mapping workflow for a single polygon (part).
    Returns:
        GeoDataFrame dissolved by 'severity' (with 'severity' column),
        reprojected to 'EPSG:4283', or None if nothing to save.
    """
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
        print("No baseline data for this part. Skipping.")
        return None
    closest_bl = baseline.isel(time=-1)

    post = load_ard_with_fallback(dc, gpgon, time=(start_date_post, end_date_post),
                                  min_gooddata_thresholds=(0.90,))
    if post.time.size == 0:
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

    # --- Vectorize (exclude unburnt class 0)
    print("Vectorizing severity raster (per-part)...")
    severity_vectors = xr_vectorize(final_severity,
                                    attribute_col='severity',
                                    crs=OUTPUT_CRS,
                                    mask=final_severity != 0)
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

    # --- Optional per-part saves
    if save_per_part_vectors:
        out_vec = os.path.join(OUTPUT_PRODUCT_DIR, f'burn_severity_polygons_{fire_name_part}.geojson')
        aggregated.to_file(out_vec, driver='GeoJSON')
        print(f"Saved per-part severity GeoJSON: {out_vec}")

    if save_per_part_rasters:
        out_cog_preview = os.path.join(OUTPUT_PRODUCT_DIR, f's2_postfire_preview_{fire_name_part}.tif')
        write_cog(post.isel(time=0).to_array().compute(), fname=out_cog_preview, overwrite=True)
        print(f"Saved post-fire preview COG: {out_cog_preview}")

        out_cog_debug = os.path.join(OUTPUT_PRODUCT_DIR, f'debug_mask_raster_{fire_name_part}.tif')
        write_cog(debug_mask.compute(), fname=out_cog_debug, overwrite=True)
        print(f"Saved debug mask COG: {out_cog_debug}")

    print(f"Successfully processed part: {fire_name_part}")
    return aggregated


# =========================
# ========= MAIN ==========
# =========================

def main():
    os.makedirs(OUTPUT_PRODUCT_DIR, exist_ok=True)
    print(f"All outputs will be saved to: {OUTPUT_PRODUCT_DIR}")

    dc = datacube.Datacube(app="Burnt_Area_Mapping")

    all_polys = load_and_prepare_polygons(POLYGON_PATH)
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

        combined_path = os.path.join(OUTPUT_PRODUCT_DIR, f"burn_severity_polygons_{base_fire_slug}.geojson")
        if SAVE_COMBINED_PER_FIRE_GEOJSON and os.path.exists(combined_path):
            print(f"[Group '{base_fire_name}'] Combined GeoJSON exists. Skipping combined write.")
            combined_skip += 1
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
                    save_per_part_rasters=SAVE_PER_PART_RASTERS
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

                    combined_gdf.to_file(combined_path, driver="GeoJSON")
                    print(f"[COMBINED] Saved MultiPolygon GeoJSON: {combined_path}")
                    combined_success += 1
                except Exception as e:
                    print(f"!!! FAILED to save combined GeoJSON for '{base_fire_name}': {e}")
                    traceback.print_exc()
            else:
                print(f"No per-part severity vectors to combine for '{base_fire_name}'.")

    print("\n" + "="*80)
    print("Batch processing complete.")
    print(f"  Per-part success: {part_success}")
    print(f"  Per-part failed: {part_fail}")
    if SAVE_COMBINED_PER_FIRE_GEOJSON:
        print(f"  Combined (group) success: {combined_success}")
        print(f"  Combined (group) skipped (exists): {combined_skip}")
    print("="*80)


if __name__ == "__main__":
    main()
