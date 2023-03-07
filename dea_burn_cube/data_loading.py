import logging
from typing import List, Tuple

import datacube
import dea_tools.datahandling
import geopandas as gpd
import s3fs
import xarray as xr

import dea_burn_cube.utils as utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


class IncorrectInputDataError(Exception):
    def __init__(self, message):
        super().__init__(message)

    def log_error(self):
        logger.error(str(self))


def _get_gpgon(region_id: str) -> datacube.utils.geometry.Geometry:
    """
    Get a geometry that covers the specified region for use with datacube.load().

    Parameters
    ----------
    region_id : str
        The ID of the region to get a geometry for. E.g. x30y29

    Returns
    -------
    gpgon : datacube.utils.geometry.Geometry object
        The geometry object representing the region specified by `region_id`.
    """
    # Use region_id to query AU-30 grid file, and get its geometry
    _ = s3fs.S3FileSystem(anon=True)

    _ = "s3" in gpd.io.file._VALID_URLS
    gpd.io.file._VALID_URLS.discard("s3")

    # Read in the AU-30 grid file
    au_grid = gpd.read_file(
        "s3://dea-public-data-dev/projects/burn_cube/configs/au-grid.geojson"
    )

    # Filter the grid to only include the specified region
    au_grid = au_grid.to_crs(epsg="3577")
    au_grid = au_grid[au_grid["region_code"] == region_id]

    # Create a Geometry object from the selected region
    gpgon = datacube.utils.geometry.Geometry(
        au_grid.geometry[au_grid.index[0]], crs="epsg:3577"
    )

    # Return the resulting Geometry object
    return gpgon


@utils.log_execution_time
def check_input_datasets(
    hnrs_dc: datacube.Datacube,
    odc_dc: datacube.Datacube,
    period: Tuple[str, str],
    mapping_period: Tuple[str, str],
    geomed_product_name: str,
    wofs_summary_product_name: str,
    ard_product_names: List[str],
    region_id: str,
) -> datacube.utils.geometry.Geometry:
    """
    Check the input datasets and get the geometry.

    Args:
        hnrs_dc : datacube.Datacube
            The Datacube object for the hnrs DC.
        odc_dc : datacube.Datacube
            The Datacube object for the ODC DC.
        period : tuple of str
            A list of two strings representing the start and end dates of the period to query for GeoMAD.
        mapping_period : tuple of str
            A list of two strings representing the start and end dates of the period to query for WOfS summary product.
        geomed_product_name : str
            The name of the GeoMAD product.
        wofs_summary_product_name : str
            The name of the WOfS summary product.
        ard_product_names: List[str]
            The names of the ARD products
        region_id : str
            The ID of the region to get a geometry for. E.g. x30y29

    Returns:
        datacube.utils.geometry.Geometry: The geometry of the region.

    Raises:
        IncorrectInputDataError:
            If there is no or more than one dataset found for either GeoMAD or WOfS summary product.

    """

    gpgon = _get_gpgon(region_id)

    # Use find_datasets to get the GeoMAD dataset ID, and display it on LOG
    datasets = odc_dc.find_datasets(
        product=wofs_summary_product_name, geopolygon=gpgon, time=mapping_period[0]
    )

    logger.info("Load WOfS summary product from %s", geomed_product_name)

    # clean up the dataset by region_code
    datasets = [
        e
        for e in datasets
        if e.metadata_doc["properties"]["odc:region_code"] == region_id
    ]

    for dataset in datasets:
        logger.info(
            "Find WOfS summary dataset with metadata:  %s",
            dataset.metadata_doc["label"],
        )

    if len(datasets) == 0:
        raise IncorrectInputDataError("Cannot find WOfS summary dataset")
    elif len(datasets) > 1:
        raise IncorrectInputDataError("Find one more than WOfS summary dataset")

    # Use find_datasets to get the ARD dataset ID, and display it on LOG
    datasets = odc_dc.find_datasets(
        product=ard_product_names, geopolygon=gpgon, time=mapping_period[0]
    )

    if len(datasets) == 0:
        raise IncorrectInputDataError("Cannot find any ARD dataset")

    # Use find_datasets to get the GeoMAD dataset ID, and display it on LOG
    datasets = hnrs_dc.find_datasets(
        product=geomed_product_name, geopolygon=gpgon, time=period[0]
    )

    logger.info("Load GeoMAD from %s", geomed_product_name)

    # clean up the dataset by region_code
    datasets = [
        e
        for e in datasets
        if e.metadata_doc["properties"]["odc:region_code"] == region_id
    ]

    for dataset in datasets:
        logger.info(
            "Find GeoMAD dataset with metadata:  %s", dataset.metadata_doc["label"]
        )

    # Ideally, the number of datasets should be 1
    if len(datasets) == 0:
        raise IncorrectInputDataError("Cannot find GeoMAD dataset")
    elif len(datasets) > 1:
        raise IncorrectInputDataError("Find one more than GeoMAD dataset")

    # Load the geometry from OpenDataCube again, avoid the pixel mismatch issue
    geometry_list = [datasets[0].extent]

    region_polygon = gpd.GeoDataFrame(
        index=range(len(geometry_list)), crs="epsg:3577", geometry=geometry_list
    )
    gpgon = datacube.utils.geometry.Geometry(
        region_polygon.geometry[0], crs="epsg:3577"
    )

    return gpgon


@utils.log_execution_time
def load_geomed_ds(
    hnrs_dc: datacube.Datacube,
    gpgon: datacube.utils.geometry.Geometry,
    period: Tuple[str, str],
    geomed_product_name: str,
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
        geomed_product_name : str
            The name of the GeoMAD product to be loaded.
        geomed_bands : list of str
            A list of strings representing the names of the bands to be loaded.

    Returns:
        xr.Dataset:
            A xarray dataset containing the loaded GeoMAD data.

    """
    geomed = hnrs_dc.load(
        geomed_product_name,
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
    ard_product_names: List[str],
    ard_bands: List[str],
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
        ard_product_names : List[str]
            Names of the ARD products to load.
        ard_bands : List[str]
            Names of the measurement bands to load.

    Returns:
        xr.Dataset: A dataset containing the loaded ARD data.
    """
    ard = dea_tools.datahandling.load_ard(
        dc=odc_dc,
        products=ard_product_names,
        measurements=ard_bands,
        geopolygon=gpgon,
        output_crs="EPSG:3577",
        resolution=(-30, 30),
        resampling={"fmask": "nearest", "*": "bilinear"},
        # mask_filters=[("dilation", 10)],
        # mask_contiguity=True,
        dask_chunks={},
        # predicate=gqa_predicate,
        time=period,
        group_by="solar_day",
    )

    ard = ard[ard_bands].to_array(dim="band").to_dataset(name="ard")

    return ard


@utils.log_execution_time
def load_wofs_summary_ds(
    odc_dc: datacube.Datacube,
    gpgon: datacube.utils.geometry.Geometry,
    mappingperiod: Tuple[str, str],
    wofs_summary_product_name: str,
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
        wofs_summary_product_name : str
            Name of the WOfS summary product to load.

    Returns:
        xr.Dataset: The loaded WOfS summary data.
    """
    wofs_summary = odc_dc.load(
        wofs_summary_product_name,
        time=mappingperiod[0],
        geopolygon=gpgon,
        resolution=(-30, 30),
        dask_chunks={},
    )

    return wofs_summary


@utils.log_execution_time
def load_reference_data(
    odc_dc: datacube.Datacube,
    hnrs_dc: datacube.Datacube,
    ard_product_names: List[str],
    geomed_product_name: str,
    ard_bands: List[str],
    geomed_bands: List[str],
    period: Tuple[str, str],
    gpgon: datacube.utils.geometry.Geometry,
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Load reference data for a given time period and spatial extent.

    Args:
        odc_dc : datacube.Datacube
            Datacube object for loading ARD data.
        hnrs_dc : datacube.Datacube
            Datacube object for loading geomedian data.
        ard_product_names : List[str]
            List of product names to use for loading ARD data.
        geomed_product_name : str
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

    Returns:
        Tuple[xr.Dataset, xr.Dataset]: A tuple of two xarray datasets containing the loaded ARD
            and geomedian data, respectively.
    """

    # Load ARD data
    ard = load_ard_ds(
        odc_dc,
        gpgon,
        period,
        ard_product_names,
        ard_bands,
    )

    ard = ard.ard

    # Load geomedian data
    geomed = load_geomed_ds(hnrs_dc, gpgon, period, geomed_product_name, geomed_bands)

    geomed = geomed.geomedian

    # Load the data into memory
    geomed = geomed.load()
    ard = ard.load()

    return ard, geomed


@utils.log_execution_time
def load_mapping_data(
    odc_dc: datacube.Datacube,
    ard_product_names: List[str],
    ard_bands: List[str],
    mappingperiod: Tuple[str, str],
    gpgon: datacube.utils.geometry.Geometry,
) -> xr.Dataset:
    """
    Loads and returns mapping data as an xarray.Dataset.

    Parameters
    ----------
    odc_dc : datacube.Datacube
        Datacube object for loading data.
    ard_product_names : List[str]
        A list of product names to load as ARD.
    ard_bands : List[str]
        A list of band names to load.
    mappingperiod : Tuple[str, str]
        A tuple containing the start and end date of the time period of interest, formatted as "YYYY-MM-DD".
    gpgon : datacube.utils.geometry.Geometry
        A geometry defining the spatial region of interest.

    Returns
    -------
    xr.Dataset
        An xarray dataset containing the loaded mapping data.
    """
    mapping_ard = load_ard_ds(
        odc_dc,
        gpgon,
        mappingperiod,
        ard_product_names,
        ard_bands,
    )

    mapping_ard = mapping_ard.ard

    mapping_ard = mapping_ard.load()

    return mapping_ard
