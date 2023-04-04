"""
This module contains functions and classes for burn mapping using data from Digital Earth
Australia (DEA).

This module contains functions and variables for processing data from a burn cube.

"""


import logging
import os
from typing import List, Tuple

import datacube
import geopandas as gpd
import rasterio.features
import xarray as xr
from shapely.ops import unary_union

from dea_burn_cube import algo, bc_data_loading, task

logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


os.environ["SQLALCHEMY_SILENCE_UBER_WARNING"] = "1"


@task.log_execution_time
def generate_ocean_mask(ds: xr.Dataset, region_id: str) -> xr.DataArray:
    """
    Generate a mask for non-ocean areas within a given region.

    Parameters
    ----------
    ds : xarray.Dataset
        A dataset containing geospatial data for the given region.
    region_id : str
        A string representing the region ID to use when selecting the grid cells
        for the given region.

    Returns
    -------
    xr.DataArray
        A 2D array of boolean values indicating whether each cell is not part of the
        ocean (True) or is part of the ocean (False).
    """
    # Load the grid for the given region
    au_grid: gpd.GeoDataFrame = gpd.read_file(
        "s3://dea-public-data-dev/projects/burn_cube/configs/au-grid.geojson"
    )
    au_grid = au_grid.to_crs(epsg="3577")
    au_grid = au_grid[au_grid["region_code"] == region_id]

    # Load the coastline shapefile for masking the ocean
    ancillary_folder: str = "s3://dea-public-data-dev/projects/burn_cube/configs"
    ocean_mask_path: str = f"{ancillary_folder}/ITEMCoastlineCleaned.shp"

    ocean_df = gpd.read_file(ocean_mask_path)
    ocean_mask = unary_union(list(ocean_df.geometry))

    land_area = au_grid.geometry[au_grid.index[0]].intersection(ocean_mask)

    y, x = ds.geobox.shape
    transform = ds.geobox.transform
    dims = ds.geobox.dims

    xy_coords = [ds[dims[0]], ds[dims[1]]]

    arr = rasterio.features.rasterize(
        shapes=[land_area], out_shape=(y, x), transform=transform
    )

    not_ocean_layer = xr.DataArray(
        arr, coords=xy_coords, dims=dims, name="not_ocean_layer"
    )
    data = xr.combine_by_coords(
        [ds, not_ocean_layer], coords=["x", "y"], join="inner", combine_attrs="override"
    )
    return data.not_ocean_layer


@task.log_execution_time
def apply_post_processing_by_wo_summary(
    odc_dc, burn_cube_result, gpgon, mappingperiod, wofs_summary_product_name
):
    """
    Applies post-processing to the given burn cube result dataset based on water observations from space (WOfS)
    summary data.

    Parameters:
    -----------
    odc_dc : datacube.Datacube
        An instance of `datacube.Datacube`.
    burn_cube_result : xarray.Dataset
        The burn cube result dataset.
    gpgon : str
        The geographic area of interest, as a string. Example: "Global".
    mappingperiod : str
        The mapping period, as a string. Example: "20200301-20200430".
    wofs_summary_product_name : str
        The name of the WOfS summary product.

    Returns:
    --------
    xarray.Dataset
            The burn cube result dataset with post-processing applied based on WOfS summary data.
    """

    wofs_summary = bc_data_loading.load_wofs_summary_ds(
        odc_dc, gpgon, mappingperiod, wofs_summary_product_name
    )

    wofs_summary_frequency = wofs_summary.frequency

    wofs_summary_frequency = wofs_summary_frequency.load()

    # Create a water mask by identifying areas with water frequency greater than or equal to 0.2
    water_mask = (wofs_summary_frequency[0, :, :].values >= 0.2).astype(float)

    # Create the opposite mask by inverting the boolean values of the water mask
    not_water_mask = (~water_mask.astype(bool)).astype(float)

    # ocean_mask = generate_ocean_mask(wofs_summary_frequency, region_id)

    burnpixel_mod = algo.burnpixel_masking(
        burn_cube_result, "Moderate"
    )  # mask the burnt area with "Medium" burnt area
    burnpixel_sev = algo.burnpixel_masking(burn_cube_result, "Severe")

    wofs_moderate = not_water_mask * burnpixel_mod
    wofs_severe = not_water_mask * burnpixel_sev
    wofs_severity = not_water_mask * burn_cube_result["Severity"]
    wofs_startdate = not_water_mask * burn_cube_result["StartDate"]
    wofs_duration = not_water_mask * burn_cube_result["Duration"]
    wofs_corroborate = not_water_mask * burn_cube_result["Corroborate"]
    wofs_cleaned = not_water_mask * burn_cube_result["Cleaned"]

    # ocean_moderate = ocean_mask * wofs_moderate
    # ocean_severe = ocean_mask * wofs_severe
    # ocean_severity = ocean_mask * wofs_severity
    # ocean_startdate = ocean_mask * wofs_startdate
    # ocean_duration = ocean_mask * wofs_duration
    # ocean_corroborate = ocean_mask * wofs_corroborate
    # ocean_cleaned = ocean_mask * wofs_cleaned

    return xr.Dataset(
        {
            "StartDate": burn_cube_result["StartDate"],
            "Duration": burn_cube_result["Duration"],
            "Severity": burn_cube_result["Severity"],
            "Severe": burn_cube_result["Severe"],
            "Moderate": burn_cube_result["Moderate"],
            "Corroborate": burn_cube_result["Corroborate"],
            "Cleaned": burn_cube_result["Cleaned"],
            "Count": burn_cube_result["Count"],
            "WOfSModerate": wofs_moderate,
            "WOfSSevere": wofs_severe,
            "WOfSSeverity": wofs_severity,
            "WOfSStartDate": wofs_startdate,
            "WOfSDuration": wofs_duration,
            "WOfSCorroborate": wofs_corroborate,
            "WOfSCleaned": wofs_cleaned,
            # "OceanModerate": ocean_moderate,
            # "OceanSevere": ocean_severe,
            # "OceanSeverity": ocean_severity,
            # "OceanStartDate": ocean_startdate,
            # "OceanDuration": ocean_duration,
            # "OceanCorroborate": ocean_corroborate,
            # "OceanCleaned": ocean_cleaned,
        }
    )


@task.log_execution_time
def generate_reference_result(
    ard: xr.Dataset, geomed: xr.Dataset, n_procs: int
) -> xr.Dataset:
    """
    Generates a reference result by computing the outliers between the input
    `ard` and `geomed` datasets using the `distances` and `outliers` functions
    from the `algo` module.

    Parameters
    ----------
    ard : xr.Dataset
        The input dataset that represents the acquired data.
    geomed : xr.Dataset
        The input dataset that represents the geometric median of the
        acquired data.
    n_procs : int
        The size of process pool

    Returns
    -------
    xr.Dataset
        The output dataset that represents the outliers between the input
        `ard` and `geomed` datasets.
    """
    dis = algo.distances(ard, geomed, n_procs)
    outliers_result = algo.outliers(ard, dis)
    return outliers_result


@task.log_execution_time
def generate_bc_result(
    odc_dc: datacube.Datacube,
    hnrs_dc: datacube.Datacube,
    ard_product_names: List[str],
    geomed_product_name: str,
    ard_bands: List[str],
    geomed_bands: List[str],
    period: Tuple[str, str],
    mappingperiod: Tuple[str, str],
    gpgon: datacube.utils.geometry.Geometry,
    task_id: str,
    output: str,
    n_procs: int,
    platform: str,
) -> xr.Dataset:
    """
    Generate burnt area severity mapping result for a given period of time.

    Parameters
    ----------
    odc_dc : datacube.Datacube
        Datacube object for loading mapping data.
    hnrs_dc : datacube.Datacube
        Datacube object for loading reference data.
    ard_product_names : list of str
        List of names of Analysis Ready Data (ARD) products.
    geomed_product_name : str
        Name of geomedian product.
    ard_bands : list of str
        List of measurement names to load for ARD products.
    geomed_bands : list of str
        List of measurement names to load for geomedian product.
    period : tuple of str
        Start and end dates of the reference data period in the format "YYYY-MM-DD".
    mappingperiod : tuple of str
        Start and end dates of the mapping data period in the format "YYYY-MM-DD".
    gpgon : datacube.utils.geometry.Geometry
        Geopolygon to load data for.
    task_id : str
        Identifier for the task being executed.
    output : str
        Path to the output directory.
    n_procs : int
        The size of process pool.
    platform : str
            The platform of data. E.g. s2 or ls.

    Returns
    -------
    xr.Dataset
        Burnt area severity mapping result.
    """

    logger.info("Begin to load reference data")
    ard, geomed = bc_data_loading.load_reference_data(
        odc_dc,
        hnrs_dc,
        ard_product_names,
        geomed_product_name,
        ard_bands,
        geomed_bands,
        period,
        gpgon,
        platform,
    )

    outliers_result = generate_reference_result(ard, geomed, n_procs)

    del ard

    logger.info("Begin to load mapping data")
    mapping_ard = bc_data_loading.load_mapping_data(
        odc_dc,
        ard_product_names,
        ard_bands,
        mappingperiod,
        gpgon,
        platform,
    )

    mapping_dis = algo.distances(mapping_ard, geomed, n_procs)

    hotspot_csv_file = f"{task_id}-hotspot_historic.csv"

    # the hotspotfile setup will be finished by step: update_hotspot_data
    hotspotfile = f"{output}/ancillary_file/{hotspot_csv_file}"

    logger.info("Load hotspot information from:  %s", hotspotfile)

    severitymapping_result = algo.severitymapping(
        mapping_dis,
        outliers_result,
        mappingperiod,
        hotspotfile,
        method="NBRdist",
        growing=True,
        n_procs=n_procs,
    )

    return severitymapping_result
