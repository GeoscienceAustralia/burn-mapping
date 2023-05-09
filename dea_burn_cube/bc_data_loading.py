"""
This module contains functions and classes for burn mapping using data from Digital Earth
Australia (DEA).

This module contains functions and variables for loading data from a burn cube.

"""

import logging
from typing import List, Tuple

import datacube
import dea_tools.datahandling
import xarray as xr

from dea_burn_cube import helper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


@helper.log_execution_time
def load_geomed_ds(
    hnrs_dc: datacube.Datacube,
    gpgon: datacube.utils.geometry.Geometry,
    period: Tuple[str, str],
    geomed_name: str,
    geomed_bands: List[str],
) -> xr.Dataset:
    """
    Load the GeoMAD data for a given region and time period.

    Args:
        hnrs_dc : datacube.Datacube
            Datacube object for loading data.
        gpgon : datacube.utils.geometry.Geometry
            A geometry that represents the region of interest.
        period : tuple of str
            A tuple of two strings representing the start and end date of the time period.
        geomed_name : str
            The name of the GeoMAD product to be loaded.
        geomed_bands : list of str
            A list of strings representing the names of the bands to be loaded.

    Returns:
        xr.Dataset:
            A xarray dataset containing the loaded GeoMAD data.

    """
    geomed = hnrs_dc.load(
        geomed_name,
        time=period[0],
        geopolygon=gpgon,
        output_crs="EPSG:3577",
        resolution=(-30, 30),
        dask_chunks={},
    )

    geomed = geomed[geomed_bands].to_array(dim="band").to_dataset(name="geomedian")

    return geomed


def load_ard_ds(
    odc_dc: datacube.Datacube,
    gpgon: datacube.utils.geometry.Geometry,
    period: Tuple[str, str],
    ard_names: List[str],
    ard_bands: List[str],
    platform: str,
) -> xr.Dataset:
    """
    Load Analysis Ready Data (ARD) for a given time period and spatial extent.

    Args:
        odc_dc : datacube.Datacube
            Datacube object for loading data.
        gpgon : datacube.utils.geometry.Geometry
            Spatial extent for loading data.
        period : Tuple[str, str]
            Start and end dates for the data, in "YYYY-MM-DD" format.
        ard_names : List[str]
            Names of the ARD products to load.
        ard_bands : List[str]
            Names of the measurement bands to load.
        platform : str
            The platform of data. E.g. s2 or ls

    Returns:
        xr.Dataset: A dataset containing the loaded ARD data.
    """

    if platform == "s2":
        cloud_mask = "s2cloudless"
    else:
        cloud_mask = "fmask"

    ard = dea_tools.datahandling.load_ard(
        dc=odc_dc,
        products=ard_names,
        cloud_mask=cloud_mask,
        measurements=ard_bands,
        geopolygon=gpgon,
        output_crs="EPSG:3577",
        resolution=(-30, 30),
        dask_chunks={},
        time=period,
        group_by="solar_day",
    )

    ard = ard[ard_bands].to_array(dim="band").to_dataset(name="ard")

    return ard


@helper.log_execution_time
def load_wofs_summary_ds(
    odc_dc: datacube.Datacube,
    gpgon: datacube.utils.geometry.Geometry,
    mappingperiod: Tuple[str, str],
    wofs_summary_name: str,
) -> xr.Dataset:
    """
    Load the WOfS summary data for a given region and time period.

    Args:
        odc_dc : datacube.Datacube
            Datacube object for loading data.
        gpgon : datacube.utils.geometry.Geometry
            Polygon of region of interest.
        mappingperiod : Tuple[str, str]
            Time period to load data for.
        wofs_summary_name : str
            Name of the WOfS summary product to load.

    Returns:
        xr.Dataset: The loaded WOfS summary data.
    """
    wofs_summary = odc_dc.load(
        wofs_summary_name,
        time=mappingperiod[0],
        geopolygon=gpgon,
        resolution=(-30, 30),
        dask_chunks={},
    )

    return wofs_summary


@helper.log_execution_time
def load_reference_data(
    odc_dc: datacube.Datacube,
    hnrs_dc: datacube.Datacube,
    ard_names: List[str],
    geomed_name: str,
    ard_bands: List[str],
    geomed_bands: List[str],
    period: Tuple[str, str],
    gpgon: datacube.utils.geometry.Geometry,
    platform: str,
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Load reference data for a given time period and spatial extent.

    Args:
        odc_dc : datacube.Datacube
            Datacube object for loading ARD data.
        hnrs_dc : datacube.Datacube
            Datacube object for loading geomedian data.
        ard_names : List[str]
            List of product names to use for loading ARD data.
        geomed_name : str
            Name of the product to use for loading geomedian data.
        ard_bands : List[str]
            List of band names to load from ARD data.
        geomed_bands : List[str]
            List of band names to load from geomedian data.
        period: Tuple[str, str])
            A tuple of two strings indicating the start and end dates (in "YYYY-MM-DD" format) for the
            time period of interest.
        gpgon : datacube.utils.geometry.Geometry
            A polygon object defining the spatial extent of interest.
        platform : str
            The platform of data. E.g. s2 or ls

    Returns:
        Tuple[xr.Dataset, xr.Dataset]: A tuple of two xarray datasets containing the loaded ARD
            and geomedian data, respectively.
    """

    # Load ARD data
    ard = load_ard_ds(
        odc_dc,
        gpgon,
        period,
        ard_names,
        ard_bands,
        platform,
    )

    ard = ard.ard

    # Load geomedian data
    geomed = load_geomed_ds(hnrs_dc, gpgon, period, geomed_name, geomed_bands)

    geomed = geomed.geomedian

    # Load the data into memory
    geomed = geomed.load()
    ard = ard.load()

    return ard, geomed


@helper.log_execution_time
def load_mapping_data(
    odc_dc: datacube.Datacube,
    ard_names: List[str],
    ard_bands: List[str],
    mappingperiod: Tuple[str, str],
    gpgon: datacube.utils.geometry.Geometry,
    platform: str,
) -> xr.Dataset:
    """
    Loads and returns mapping data as an xarray.Dataset.

    Parameters
    ----------
    odc_dc : datacube.Datacube
        Datacube object for loading data.
    ard_names : List[str]
        A list of product names to load as ARD.
    ard_bands : List[str]
        A list of band names to load.
    mappingperiod : Tuple[str, str]
        A tuple containing the start and end date of the time period of interest, formatted as "YYYY-MM-DD".
    gpgon : datacube.utils.geometry.Geometry
        A geometry defining the spatial region of interest.
    platform : str
        The platform of data. E.g. s2 or ls

    Returns
    -------
    xr.Dataset
        An xarray dataset containing the loaded mapping data.
    """
    mapping_ard = load_ard_ds(
        odc_dc,
        gpgon,
        mappingperiod,
        ard_names,
        ard_bands,
        platform,
    )

    mapping_ard = mapping_ard.ard

    mapping_ard = mapping_ard.load()

    return mapping_ard
