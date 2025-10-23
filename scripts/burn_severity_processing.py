#!/usr/bin/env python3
"""
Burnt Severity Classification Script

This script processes Sentinel-2 ARD (Analysis Ready Data) to map the extent
and severity of a bushfire. It compares a pre-fire baseline image with a
post-fire time series to calculate the delta Normalized Burn Ratio (dNBR).

It classifies severity differently for woody vs. grassy landcover types,
masks out bad data (clouds, water), and saves the final classification
as both a GeoJSON shapefile (one per fire) and preview GeoTIFFs (one per
polygon part).
"""

# Standard library imports
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import traceback # For detailed error logging in main loop

# Third-party imports
import datacube
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from affine import Affine
from datacube.utils.cog import write_cog
from datacube.utils.geometry import CRS, Geometry
from scipy import ndimage
from skimage import morphology

# Local/custom tool imports
# Assumes 'dea-tools' is in a relative path
sys.path.insert(1, '../../Temp_branch/dea-notebooks/Tools')
try:
    from dea_tools.datahandling import load_ard
    from dea_tools.plotting import display_map, rgb
    from dea_tools.bandindices import calculate_indices
    from dea_tools.spatial import xr_vectorize
except ImportError as e:
    print(f"Error: Could not import 'dea-tools'.")
    print("Please ensure the path in 'sys.path.insert' is correct.")
    print(f"Details: {e}")
    sys.exit(1)

# --- Constants ---

# Input data
POLYGON_PATH = '10k_sampled_fires.geojson'

# --- NEW: Batch processing setting ---
MAX_POLYGONS_TO_PROCESS = 10  # Process the first N fires from the file

# Output directory
OUTPUT_PRODUCT_DIR = 'products'

# Datacube parameters
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


def load_and_prepare_polygons(path: str) -> gpd.GeoDataFrame | None:
    """
    Loads the fire polygon GeoJSON and prepares it for processing.
    
    This includes dissolving by 'fire_id' to ensure one row per fire.
    
    Args:
        path: File path to the GeoJSON.

    Returns:
        A GeoDataFrame, or None if the file is not found.
    """
    print(f"Loading all polygons from: {path}")
    try:
        poly_gdf = gpd.read_file(path)
    except FileNotFoundError:
        print(f"Error: Input polygon file not found at {path}")
        return None
    
    # Dissolve if multiple polygons (e.g., from a sample)
    try:
        if len(poly_gdf) > 1 and 'fire_id' in poly_gdf.columns:
            print("Dissolving polygons by 'fire_id'...")
            poly_gdf = poly_gdf.dissolve(by='fire_id')
        elif 'fire_id' not in poly_gdf.columns:
            print("Warning: 'fire_id' not in columns. Not dissolving.")
    except TypeError as e:
        print(f"Warning: Could not dissolve polygon ({e}). Continuing with loaded data.")
        pass

    # Copy index (which is 'fire_id' after dissolve) to a column
    # This is for metadata; the geometry is the important part
    if 'fire_id' not in poly_gdf.columns:
         poly_gdf["fire_id"] = list(poly_gdf.index)
    
    return poly_gdf


def load_ard_with_fallback(dc: datacube.Datacube, 
                           gpgon: Geometry, 
                           time: tuple, 
                           min_gooddata_thresholds: list = [0.99, 0.90],
                           **kwargs) -> xr.Dataset:
    """
    Loads ARD data, trying a list of 'min_gooddata' thresholds in order.
    
    This avoids duplicating the 'load_ard' call and ensures data is
    loaded if the primary threshold (e.g., 99%) finds no images.
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
    
    for threshold in min_gooddata_thresholds:
        print(f"Attempting to load data with min_gooddata = {threshold}...")
        base_params['min_gooddata'] = threshold
        data = load_ard(**base_params)
        
        if data.time.size > 0:
            print(f"Success: Loaded {data.time.size} time-slices.")
            return data
            
    print(f"Warning: No data found for time range {time} "
          f"even with min_gooddata = {min_gooddata_thresholds[-1]}")
    return data


def calculate_severity(delta_nbr: xr.DataArray, 
                       landcover: xr.DataArray,
                       grass_classes: list) -> xr.DataArray:
    """
    Calculates burn severity using different thresholds for woody vs. grass.
    """
    print("Calculating severity based on landcover...")
    
    # 1. Create a binary grass mask
    grass_mask = landcover.level4.isin(grass_classes)

    # 2. Calculate woody severity
    sev_woody = xr.zeros_like(delta_nbr.NBR, dtype=np.uint8)
    sev_woody = xr.where(delta_nbr.NBR >= 0.1,  2, sev_woody) # Low
    sev_woody = xr.where(delta_nbr.NBR >= 0.27, 3, sev_woody) # Medium
    sev_woody = xr.where(delta_nbr.NBR >= 0.44, 4, sev_woody) # High
    sev_woody = xr.where(delta_nbr.NBR >= 0.66, 5, sev_woody) # Very High
    
    severity_woody_masked = sev_woody.where(grass_mask == 0, 0)

    # 3. Calculate grass severity
    severity_grass = grass_mask.where(delta_nbr.NBR >= 0.1, 0)

    # 4. Combine
    severity = severity_woody_masked + severity_grass
    severity.name = 'severity'
    return severity


def create_debug_mask(pre_fire_scene: xr.Dataset, 
                      post_fire_stack: xr.Dataset) -> xr.DataArray:
    """
    Creates a mask for pixels to be excluded from the analysis.
    """
    print("Creating debug/masking layer...")
    
    debug_layer_blank = xr.ones_like(pre_fire_scene.nbart_red, dtype=np.uint16)
    
    # --- 1. Water Mask (Post-fire) ---
    post_mndwi = calculate_indices(post_fire_stack, 
                                   index='MNDWI', 
                                   collection='ga_s2_3', 
                                   drop=True)
    max_mndwi = post_mndwi.max('time')
    post_water = debug_layer_blank.where(max_mndwi.MNDWI > 0, 0) # Class 1
    new_debug = post_water

    # --- 2. Pre-fire Cloud Mask ---
    pre_cloud = (debug_layer_blank.where(
        pre_fire_scene.oa_s2cloudless_mask == 2, 0)) * 10 # Class 10
    new_debug = new_debug + pre_cloud

    # --- 3. Post-fire Persistent Cloud Mask ---
    post_cloud = post_fire_stack.oa_s2cloudless_mask.where(
        post_fire_stack.oa_s2cloudless_mask >= 1, 1)
    persistent_cloud = post_cloud.min('time') 
    post_cloud_mask = (debug_layer_blank.where(
        persistent_cloud == 2, 0)) * 100 # Class 100
    new_debug = new_debug + post_cloud_mask
    
    # --- 4. Pre-fire Contiguity Mask ---
    pre_contiguity = (debug_layer_blank.where(
        pre_fire_scene.oa_nbart_contiguity != 1, 0)) * 1000 # Class 1000
    new_debug = new_debug + pre_contiguity

    # --- 5. Post-fire Persistent Contiguity Mask ---
    post_contiguity = post_fire_stack.oa_nbart_contiguity.where(
        post_fire_stack.oa_nbart_contiguity == 1, 0)
    persistent_cont = post_contiguity.max('time') 
    post_contiguity_mask = (debug_layer_blank.where(
        persistent_cont != 1, 0)) * 10000 # Class 10000
    new_debug = new_debug + post_contiguity_mask
    
    new_debug.name = 'debug_mask'
    return new_debug


def process_single_fire(fire_series: pd.Series, 
                        poly_crs: CRS, 
                        dc: datacube.Datacube,
                        unique_part_name: str) -> gpd.GeoDataFrame | None:
    """
    Runs the full burn mapping workflow for a single fire polygon (part).
    
    This function now SAVES RASTER (COG) files using the `unique_part_name`
    and RETURNS the vectorized GeoDataFrame for later aggregation.
    
    Args:
        fire_series: A single row (pd.Series) containing metadata for the
                     *original fire* but the *geometry* for the *specific part*.
        poly_crs: The CRS of the input polygons.
        dc: An active Datacube instance.
        unique_part_name: A unique name for this specific polygon part,
                          used for saving RASTER files.

    Returns:
        A GeoDataFrame of the vectorized severity, or None if processing
        fails or no burn area is detected.
    """
    
    # --- 1. Extract Metadata and Geometry ---
    # fire_series.geometry is a single Polygon (a part of the original)
    gpgon = Geometry(fire_series.geometry, crs=poly_crs)
    
    # Create a single-row GeoDataFrame for clipping
    poly = gpd.GeoDataFrame([fire_series], crs=poly_crs)

    fire_id = fire_series.fire_id
    
    # 'name' is the unique name for *this part*
    name = unique_part_name

    # FIXME: 'fire_date' was used in the original script but not defined.
    # I am assuming it comes from an 'ignition_date' column.
    # Please check your GeoJSON and update this column name if incorrect.
    try:
        fire_date = str(fire_series.ignition_date)[:10]
    except AttributeError:
        print("Error: Could not find 'ignition_date' column.")
        print("Please define 'fire_date' manually or check column name.")
        raise # Re-raise to be caught by main loop

    # Safely extract extinguish date
    try:
        if pd.isna(fire_series.extinguish_date):
            extinguish_date = 'None'
        else:
            extinguish_date = str(fire_series.extinguish_date)[:10]
    except (AttributeError, KeyError):
        print("No 'extinguish' date column found. Will use default buffer.")
        extinguish_date = 'None'
    
    print(f"   Fire ID: {fire_id}")
    print(f"   Ignition: {fire_date}")
    print(f"   Extinguish: {extinguish_date}")

    # --- 2. Calculate Time Windows ---
    start_date_pre = (datetime.strptime(fire_date, '%Y-%m-%d') - 
                      timedelta(days=PRE_FIRE_BUFFER_DAYS)).strftime('%Y-%m-%d')
    end_date_pre = (datetime.strptime(fire_date, '%Y-%m-%d') - 
                    timedelta(days=1)).strftime('%Y-%m-%d')

    if extinguish_date == 'None':
        start_date_post = (datetime.strptime(fire_date, '%Y-%m-%d') + 
                           timedelta(days=POST_FIRE_START_DAYS)).strftime('%Y-%m-%d')
    else:
        start_date_post = extinguish_date
    
    end_date_post = (datetime.strptime(start_date_post, '%Y-%m-%d') + 
                     timedelta(days=POST_FIRE_WINDOW_DAYS)).strftime('%Y-%m-%d')
                     
    month_number = int(fire_date[5:7])
    if month_number >= 10:
        landcover_year = fire_date[0:4]
    else:
        landcover_year = str(int(fire_date[0:4]) - 1)

    print(f"Pre-fire window: {start_date_pre} to {end_date_pre}")
    print(f"Post-fire window: {start_date_post} to {end_date_post}")
    print(f"Landcover year: {landcover_year}")

    # --- 3. Load Data ---
    baseline = load_ard_with_fallback(dc, gpgon, 
                                      time=(start_date_pre, end_date_pre),
                                      min_gooddata_thresholds=[0.99, 0.90])
    if baseline.time.size == 0:
        print("Error: No baseline data found for this fire. Skipping.")
        return None # Skip this fire part

    closest_bl = baseline.isel(time=-1)

    post = load_ard_with_fallback(dc, gpgon, 
                                  time=(start_date_post, end_date_post),
                                  min_gooddata_thresholds=[0.90])
    if post.time.size == 0:
        print("Error: No post-fire data found for this fire. Skipping.")
        return None # Skip this fire part
        
    landcover = dc.load(
        product='ga_ls_landcover_class_cyear_3',
        geopolygon=gpgon,
        time=(landcover_year),
        output_crs=OUTPUT_CRS,
        resolution=RESOLUTION,
        group_by='solar_day',
        dask_chunks={},
    )
    if landcover.time.size == 0:
        print(f"Error: No landcover data found for year {landcover_year}. Skipping.")
        return None # Skip this fire part
    landcover = landcover.isel(time=0)

    # --- 4. Calculate Indices and Severity ---
    pre_nbr = calculate_indices(closest_bl, 
                                index='NBR', 
                                collection='ga_s2_3', 
                                drop=True)
    post_nbr = calculate_indices(post, 
                                 index='NBR', 
                                 collection='ga_s2_3', 
                                 drop=True)
    min_post_nbr = post_nbr.min('time')
    delta_nbr = pre_nbr - min_post_nbr
    severity = calculate_severity(delta_nbr, landcover, GRASS_CLASSES)

    # --- 5. Masking ---
    new_debug = create_debug_mask(closest_bl, post)
    final_severity = severity.where(new_debug == 0, 6)
    final_severity.name = 'burn_severity'

    # We explicitly copy them back from a known good source.
    # We'll use a DataArray from the 'closest_bl' dataset (e.g., nbart_red)
    # as the Dataset's (closest_bl) attrs may be empty.
    source_attrs = closest_bl.nbart_red.attrs
    

    # --- 6. Vectorize and Save ---
    print("Vectorizing severity raster...")
    severity_vectors = xr_vectorize(final_severity, 
                                    attribute_col='severity', 
                                    crs=OUTPUT_CRS,
                                    mask=final_severity != 0)
    
    if severity_vectors.empty:
        print("No burn area detected (vectors are empty). Skipping save.")
        return None # MODIFIED: Return None instead of just returning

    # Clip vectors to the precise boundary of this *part*
    clipped_severity_vectors = severity_vectors.clip(poly)
    
    if clipped_severity_vectors.empty:
        print("No burn area detected after clipping to polygon part. Skipping save.")
        return None
        
    aggregated_severity = clipped_severity_vectors.dissolve(by='severity')
    
    # Add metadata (will be used in the combined GDF)
    aggregated_severity['fire_id'] = fire_id
    aggregated_severity['fire_name'] = name # This will be the unique_fire_name
    aggregated_severity['ignition_date'] = fire_date
    aggregated_severity['extinguish_date'] = extinguish_date

    # --- MODIFIED: Save Vector Files (REMOVED) ---
    # Vector files are no longer saved here. They are returned
    # to main() to be combined.

    # --- Save Raster Files (COGs) ---
    # These *are* saved here, using the unique_part_name
    output_cog_preview = os.path.join(
        OUTPUT_PRODUCT_DIR, f's2_postfire_preview_{name}.tif')
    write_cog(post.isel(time=0).to_array().compute(), 
              fname=output_cog_preview,
              overwrite=True) # Added overwrite
    print(f"Saved post-fire preview COG to: {output_cog_preview}")
    
    output_cog_severity = os.path.join(
        OUTPUT_PRODUCT_DIR, f'burn_severity_raster_{name}.tif')
    write_cog(final_severity.compute(), 
              fname=output_cog_severity,
              overwrite=True)
    print(f"Saved severity raster COG to: {output_cog_severity}")

    output_cog_debug = os.path.join(
        OUTPUT_PRODUCT_DIR, f'debug_mask_raster_{name}.tif')
    write_cog(new_debug.compute(), 
              fname=output_cog_debug,
              overwrite=True)
    print(f"Saved debug mask raster COG to: {output_cog_debug}")
    
    print(f"Successfully processed fire part: {name}")
    
    # --- MODIFIED: Return the vector data ---
    return aggregated_severity


def main():
    """Main processing workflow to loop over fires."""
    
    # --- 0. Setup ---
    os.makedirs(OUTPUT_PRODUCT_DIR, exist_ok=True)
    print(f"All outputs will be saved to: {OUTPUT_PRODUCT_DIR}")
    
    dc = datacube.Datacube(app="Burnt_Area_Mapping")
    
    all_polys = load_and_prepare_polygons(POLYGON_PATH)
    
    if all_polys is None or all_polys.empty:
        print("No polygons loaded. Exiting.")
        return

    # --- 1. Select Fires to Process ---
    
    # Get the slice of *fires* we want to process
    num_fires_to_process = min(MAX_POLYGONS_TO_PROCESS, len(all_polys))
    print(f"Found {len(all_polys)} total fires. Selecting first {num_fires_to_process} for processing.")
    
    # Get the subset of *fires* BEFORE exploding
    polys_to_process = all_polys.iloc[:num_fires_to_process]

    success_count = 0
    fail_count = 0
    skip_count = 0  
    
    # --- 2. MODIFIED: Loop over ORIGINAL fires ---
    for original_index, original_fire_series in polys_to_process.iterrows():
        
        # --- Get original fire name (using fire_id) ---
        # We use the index, which *is* the fire_id after dissolve
        original_fire_id = original_fire_series.get('fire_id', original_index)
        original_fire_name = f"fire_id_{original_fire_id}"
        
        print("\n" + "="*80)
        print(f"Processing Fire: {original_fire_name} (Index: {original_index})")
        print("="*80)

        # --- Check if COMBINED vector file already exists ---
        output_geojson_name = os.path.join(
            OUTPUT_PRODUCT_DIR, f'burn_severity_polygons_{original_fire_name}.geojson')
        
        if os.path.exists(output_geojson_name):
            print(f"Output GeoJSON already exists: {output_geojson_name}. Skipping entire fire.")
            skip_count += 1
            continue

        # --- Prepare for part processing ---
        all_parts_vectors = [] # To store vector GDFs from each part
        fire_failed_parts = 0  # To count failed parts
        
        # Explode this single fire into its parts
        current_fire_gdf = gpd.GeoDataFrame([original_fire_series], crs=all_polys.crs)
        try:
            fire_parts_gdf = current_fire_gdf.explode(index_parts=True)
        except TypeError:
            # Fallback for older geopandas versions
            fire_parts_gdf = current_fire_gdf.explode()
        
        num_parts = len(fire_parts_gdf)
        print(f"Fire consists of {num_parts} polygon part(s).")

        # --- 3. Inner loop over each part of this fire ---
        for part_i, (part_index, part_series) in enumerate(fire_parts_gdf.iterrows()):
            
            # Create a unique name for this part's *raster* files
            if num_parts > 1:
                unique_part_name = f"{original_fire_name}_part_{part_i}"
            else:
                unique_part_name = original_fire_name
            
            print(f"\n--- Processing Part {part_i + 1}/{num_parts}: {unique_part_name} ---")

            try:
                # Create the series to process:
                # It has the *metadata* of the original fire
                # but the *geometry* of the specific part.
                processing_series = original_fire_series.copy()
                processing_series.geometry = part_series.geometry
                
                # Call the modified function. It will save rasters
                # and return the vector GDF (or None)
                part_vector_result = process_single_fire(
                    processing_series, 
                    all_polys.crs, 
                    dc, 
                    unique_part_name # This name is for the rasters
                )
                
                if part_vector_result is not None and not part_vector_result.empty:
                    all_parts_vectors.append(part_vector_result)
                else:
                    print(f"Part {unique_part_name} produced no vector data (unburnt or masked).")
                    
            except Exception as e:
                fire_failed_parts += 1
                print(f"!!! FAILED to process {unique_part_name}: {e}")
                traceback.print_exc()
                print(f"Continuing to next part, but vector output for {original_fire_name} will be incomplete.")

        # --- 4. After all parts are processed, combine vectors ---
        if fire_failed_parts > 0:
            print(f"!!! Skipping vector save for {original_fire_name} because {fire_failed_parts} part(s) failed.")
            fail_count += 1 # Mark the whole fire as failed
            continue
        
        if not all_parts_vectors:
            print(f"No burn area detected for any parts of {original_fire_name}. No vector file saved.")
            success_count += 1 # Processed successfully, just no data
            continue

        # --- 5. Combine, re-project, and save combined vector ---
        print(f"Combining {len(all_parts_vectors)} vector part(s) for {original_fire_name}...")
        try:
            combined_vectors = pd.concat(all_parts_vectors)
            
            # Re-project to the final desired CRS for vector output
            combined_vectors = combined_vectors.to_crs('EPSG:4283')
            
            # We dissolve *again* after concat to merge shapes
            # of the same severity class that might span across parts
            print("Dissolving combined vectors by severity...")
            combined_vectors = combined_vectors.dissolve(by='severity')
            
            # Re-add metadata (dissolve keeps 'severity' as index)
            combined_vectors['fire_id'] = original_fire_id
            combined_vectors['fire_name'] = original_fire_name
            try:
                combined_vectors['ignition_date'] = str(original_fire_series.ignition_date)[:10]
            except: pass
            try:
                if pd.isna(original_fire_series.extinguish_date):
                    combined_vectors['extinguish_date'] = 'None'
                else:
                    combined_vectors['extinguish_date'] = str(original_fire_series.extinguish_date)[:10]
            except:
                 combined_vectors['extinguish_date'] = 'None'


            # Save COMBINED Vector Files (using original_fire_name)
            base_output_name = os.path.join(
                OUTPUT_PRODUCT_DIR, f'burn_severity_polygons_{original_fire_name}')
            
            output_shp_name = f'{base_output_name}.shp'
            combined_vectors.to_file(output_shp_name)
            print(f"Saved COMBINED severity shapefile to: {output_shp_name}")
            
            # 'output_geojson_name' was defined earlier for the skip check
            combined_vectors.to_file(output_geojson_name, driver='GeoJSON')
            print(f"Saved COMBINED severity GeoJSON to: {output_geojson_name}")
            
            success_count += 1
            
        except Exception as e:
            fail_count += 1
            print(f"!!! FAILED to combine and save vectors for {original_fire_name}: {e}")
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("Batch processing complete.")
    print(f"  Successfully processed: {success_count} (fires)")
    print(f"  Skipped (already complete): {skip_count} (fires)")
    print(f"  Failed: {fail_count} (fires)")
    print("="*80)


if __name__ == "__main__":
    main()