"""CLI: Run Burn Cube analysis based on task_id and region_id
Geoscience Australia
2022
"""

import logging
import os
import sys
from datetime import datetime

import boto3
import click
import datacube
import dea_tools.datahandling
import geopandas as gpd
import pandas as pd
import s3fs
import xarray as xr
from datacube.utils import geometry
from datacube.utils.cog import write_cog
from shapely.ops import unary_union

import dea_burn_cube.__version__
import dea_burn_cube.utils as utils

logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

BUCKET_NAME = "dea-public-data-dev"

os.environ["SQLALCHEMY_SILENCE_UBER_WARNING"] = "1"


def gqa_predicate(ds):
    return ds.metadata.gqa_iterative_mean_xy <= 1


def display_current_step_processing_duration(
    log_text, burn_cube_process_timer, log_code
):
    # display the datacube loading time
    now = datetime.now()
    duration = now - burn_cube_process_timer

    info = f"{log_code}: {log_text} - {duration.total_seconds()} secondss"

    log_info = info

    logger.info(log_info)

    # remember to reset the timer
    return now


def get_geomed_ds(region_id, period, hnrs_config, geomed_bands, geomed_product_name):

    # Use region_id to query AU-30 grid file, and get its geometry
    _ = s3fs.S3FileSystem(anon=True)

    _ = "s3" in gpd.io.file._VALID_URLS
    gpd.io.file._VALID_URLS.discard("s3")

    au_grid = gpd.read_file(
        "s3://dea-public-data-dev/projects/burn_cube/configs/au-grid.geojson"
    )

    au_grid = au_grid.to_crs(epsg="3577")
    au_grid = au_grid[au_grid["region_code"] == region_id]

    gpgon = datacube.utils.geometry.Geometry(
        au_grid.geometry[au_grid.index[0]], crs="epsg:3577"
    )

    hnrs_dc = datacube.Datacube(app="geomed_loading", config=hnrs_config)

    # Use find_datasets to get the GeoMAD dataset ID, and display it on LOG
    datasets = hnrs_dc.find_datasets(
        product=geomed_product_name, geopolygon=gpgon, time=period[0]
    )

    # Ideally, the number of datasets should be 1
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

    # get gpgon from the clean dataset list
    # metadata = hnrs_dc.index.datasets.get(str(datasets[0].id))
    geometry_list = [datasets[0].extent]

    region_polygon = gpd.GeoDataFrame(
        index=range(len(geometry_list)), crs="epsg:3577", geometry=geometry_list
    )
    gpgon = datacube.utils.geometry.Geometry(
        region_polygon.geometry[0], crs="epsg:3577"
    )

    geomed = hnrs_dc.load(
        geomed_product_name,
        time=period[0],
        geopolygon=gpgon,
        output_crs="EPSG:3577",
        resolution=(-30, 30),
        dask_chunks={},
    )

    geomed = geomed[geomed_bands].to_array(dim="band").to_dataset(name="geomedian")

    return gpgon, geomed


def generate_ocean_mask(ds, region_id):
    au_grid = gpd.read_file(
        "s3://dea-public-data-dev/projects/burn_cube/configs/au-grid.geojson"
    )

    au_grid = au_grid.to_crs(epsg="3577")
    au_grid = au_grid[au_grid["region_code"] == region_id]

    ancillary_folder = "s3://dea-public-data-dev/projects/burn_cube/configs"
    ocean_mask_path = f"{ancillary_folder}/ITEMCoastlineCleaned.shp"

    ocean_df = gpd.read_file(ocean_mask_path)
    ocean_mask = unary_union(list(ocean_df.geometry))

    land_area = au_grid.geometry[au_grid.index[0]].intersection(ocean_mask)

    y, x = ds.geobox.shape
    transform = ds.geobox.transform
    dims = ds.geobox.dims

    xy_coords = [ds[dims[0]], ds[dims[1]]]

    import rasterio.features

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


def apply_post_processing_by_wo_summary(
    dc, burn_cube_result, gpgon, mappingperiod, x_i, y_i, split_count, region_id
):

    # TODO: we dont use Dask to walkaround the loading issue here cause the WOfS result size is small
    wofs_summary = dc.load(
        "ga_ls_wo_fq_cyear_3",
        time=mappingperiod[0],
        geopolygon=gpgon,
        resolution=(-30, 30),
        dask_chunks={},
    )

    interval = int(len(wofs_summary.frequency.x) / split_count)

    wofs_summary_frequency = wofs_summary.frequency.isel(
        x=range(x_i * interval, (x_i + 1) * interval),
        y=range(y_i * interval, (y_i + 1) * interval),
    )

    wofs_summary_frequency = wofs_summary_frequency.load()

    wofs_mask = (wofs_summary_frequency[0, :, :].values < 0.2).astype(float)

    # ocean_mask = generate_ocean_mask(wofs_summary_frequency, region_id)

    burnpixel_mod = utils.burnpixel_masking(
        burn_cube_result, "Moderate"
    )  # mask the burnt area with "Medium" burnt area
    burnpixel_sev = utils.burnpixel_masking(burn_cube_result, "Severe")

    wofs_moderate = wofs_mask * burnpixel_mod
    wofs_severe = wofs_mask * burnpixel_sev
    wofs_severity = wofs_mask * burn_cube_result["Severity"]
    wofs_startdate = wofs_mask * burn_cube_result["StartDate"]
    wofs_duration = wofs_mask * burn_cube_result["Duration"]
    wofs_corroborate = wofs_mask * burn_cube_result["Corroborate"]
    wofs_cleaned = wofs_mask * burn_cube_result["Cleaned"]

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


def generate_subregion_result(
    dc,
    geomed,
    ard_bands,
    period,
    mappingperiod,
    gpgon,
    task_id,
    region_code,
    x_i,
    y_i,
    split_count,
    burn_cube_process_timer,
    output_folder,
):

    ard = dea_tools.datahandling.load_ard(
        dc,
        products=["ga_ls8c_ard_3"],
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

    interval = int(len(ard.ard.x) / split_count)

    ard = ard.ard.isel(
        x=range(x_i * interval, (x_i + 1) * interval),
        y=range(y_i * interval, (y_i + 1) * interval),
    )
    geomed = geomed.geomedian.isel(
        x=range(x_i * interval, (x_i + 1) * interval),
        y=range(y_i * interval, (y_i + 1) * interval),
    )

    # display the Dask lazy loading time
    burn_cube_process_timer = display_current_step_processing_duration(
        log_text="Dask lazy loading duration",
        burn_cube_process_timer=burn_cube_process_timer,
        log_code=f"{task_id}-{region_code}-{x_i}-{y_i}",
    )

    geomed = geomed.load()
    ard = ard.load()

    # display the datacube loading time
    burn_cube_process_timer = display_current_step_processing_duration(
        log_text=f"The datacube loading {period} duration",
        burn_cube_process_timer=burn_cube_process_timer,
        log_code=f"{task_id}-{region_code}-{x_i}-{y_i}",
    )

    dis = utils.distances(ard, geomed)

    burn_cube_process_timer = display_current_step_processing_duration(
        log_text=f"The burn cube processing {period} distance duration",
        burn_cube_process_timer=burn_cube_process_timer,
        log_code=f"{task_id}-{region_code}-{x_i}-{y_i}",
    )

    outliers_result = utils.outliers(ard, dis)

    burn_cube_process_timer = display_current_step_processing_duration(
        log_text=f"The burn cube processing {period} outlier duration",
        burn_cube_process_timer=burn_cube_process_timer,
        log_code=f"{task_id}-{region_code}-{x_i}-{y_i}",
    )

    del ard, dis

    mapping_ard = dea_tools.datahandling.load_ard(
        dc,
        products=["ga_ls8c_ard_3"],
        measurements=ard_bands,
        geopolygon=gpgon,
        output_crs="EPSG:3577",
        resolution=(-30, 30),
        resampling={"fmask": "nearest", "*": "bilinear"},
        # mask_filters=[("dilation", 10)],
        # mask_contiguity=True,
        dask_chunks={},
        # predicate=gqa_predicate,
        time=mappingperiod,
        group_by="solar_day",
    )

    mapping_ard = mapping_ard[ard_bands].to_array(dim="band").to_dataset(name="ard")
    mapping_ard = mapping_ard.ard.isel(
        x=range(x_i * interval, (x_i + 1) * interval),
        y=range(y_i * interval, (y_i + 1) * interval),
    )

    mapping_ard = mapping_ard.load()

    burn_cube_process_timer = display_current_step_processing_duration(
        log_text=f"The datacube loading {mappingperiod} duration",
        burn_cube_process_timer=burn_cube_process_timer,
        log_code=f"{task_id}-{region_code}-{x_i}-{y_i}",
    )

    mapping_dis = utils.distances(mapping_ard, geomed)

    burn_cube_process_timer = display_current_step_processing_duration(
        log_text=f"The burn cube processing {period} distance duration",
        burn_cube_process_timer=burn_cube_process_timer,
        log_code=f"{task_id}-{region_code}-{x_i}-{y_i}",
    )

    hotspot_csv_file = f"{task_id}-hotspot_historic.csv"

    # the hotspotfile setup will be finished by step: update_hotspot_data
    hotspotfile = f"{output_folder}/ancillary_file/{hotspot_csv_file}"

    logger.info("Load hotspot information from:  %s", hotspotfile)

    severitymapping_result = utils.severitymapping(
        mapping_dis,
        outliers_result,
        mappingperiod,
        hotspotfile,
        method="NBRdist",
        growing=True,
    )

    burn_cube_process_timer = display_current_step_processing_duration(
        log_text="The burn cube processing severity_mapping_result duration",
        burn_cube_process_timer=burn_cube_process_timer,
        log_code=f"{task_id}-{region_code}-{x_i}-{y_i}",
    )

    return severitymapping_result, burn_cube_process_timer


def logging_setup(verbose: int):
    """Set up logging.

    Arguments
    ---------
    verbose : int
        Verbosity level (0, 1, 2).
    """
    loggers = [
        logging.getLogger(name)
        for name in logging.root.manager.loggerDict
        if not name.startswith("sqlalchemy") and not name.startswith("boto")
    ]
    # For compatibility with docker+pytest+click stack...
    stdout_hdlr = logging.StreamHandler(sys.stdout)
    for logger in loggers:
        if verbose == 0:
            logging.basicConfig(level=logging.WARNING)
        elif verbose == 1:
            logging.basicConfig(level=logging.INFO)
        elif verbose == 2:
            logging.basicConfig(level=logging.DEBUG)
        else:
            raise click.ClickException("Maximum verbosity is -vv")
        logger.addHandler(stdout_hdlr)
        logger.propagate = False


@click.group()
@click.version_option(version=dea_burn_cube.__version__)
def main():
    """Run dea-burn-cube."""


@main.command(no_args_is_help=True)
@click.option(
    "--task-id",
    "-t",
    type=str,
    default=None,
    help="REQUIRED. Burn Cube task id, e.g. Dec-21.",
)
@click.option(
    "--region-list-s3-path",
    "-r",
    type=str,
    default=None,
    help="REQUIRED. The AU-30 Region list in GeoJSON format.",
)
@click.option(
    "--output-s3-folder",
    "-o",
    type=str,
    default="projects/burn_cube/airflow-run/burn-cube-app/ancillary_file",
    help="The ancillary_file folder which save clean-up region list file.",
)
@click.option("-v", "--verbose", count=True)
def filter_regions(task_id, region_list_s3_path, output_s3_folder, verbose):
    """
    There are two assumptions on this method:
    1. user always use AU-30 grid standard GeoJSON
        The example region-list:
            s3://dea-public-data-dev/mangroves_aux/easter_vic.geojson
    2. user already updated hotspot, and we can use the clean-up CSV file
    """
    logging_setup(verbose)

    _ = s3fs.S3FileSystem(anon=True)

    _ = "s3" in gpd.io.file._VALID_URLS
    gpd.io.file._VALID_URLS.discard("s3")

    region_gdf = gpd.read_file(region_list_s3_path)
    region_gdf = region_gdf.to_crs(epsg="3577")

    ancillary_folder = f"s3://dea-public-data-dev/{output_s3_folder}"

    logger.info("Filter  %s  by Ocean Mask", region_list_s3_path)
    ocean_mask_path = (
        "s3://dea-public-data-dev/projects/burn_cube/configs/ITEMCoastlineCleaned.shp"
    )
    ocean_mask = gpd.read_file(ocean_mask_path)

    # the Ocean Mask CRS should be: EPSG:3577
    filter_by_ocean_mask = []

    for region_index in region_gdf.index:
        region_id = region_gdf.region_code[region_index]
        region_geometry = region_gdf.geometry[region_index]

        for ocean_index in ocean_mask.index:
            if region_geometry.intersects(ocean_mask.geometry[ocean_index]):
                filter_by_ocean_mask.append(region_id)
                break

    region_gdf = region_gdf[
        region_gdf["region_code"].isin(filter_by_ocean_mask)
    ].reindex()

    logger.info(
        "The number of region changes to %s after Ocean Mask filter",
        str(len(region_gdf)),
    )

    # we assume the formats are always same, with columns: region_code, i_x, i_y, utc_offset, geometry
    # also the geometry are always Polygon

    hotspot_file = f"{ancillary_folder}/{task_id}-hotspot_historic.csv"

    logger.info("Filter regions by Hot Spot %s", hotspot_file)

    hotspot_df = pd.read_csv(hotspot_file)

    import pyproj
    from shapely.geometry import Point
    from shapely.ops import unary_union

    latitude = hotspot_df.latitude.values
    longitude = hotspot_df.longitude.values

    reverse_transformer = pyproj.Transformer.from_crs("EPSG:4283", "EPSG:3577")
    easting, northing = reverse_transformer.transform(latitude, longitude)

    patch = [
        Point(easting[i], northing[i]).buffer(4000) for i in range(0, len(hotspot_df))
    ]
    hotspot_polygons = unary_union(patch)

    filter_by_hotspot = []

    for region_index in region_gdf.index:
        region_id = region_gdf.region_code[region_index]
        region_geometry = region_gdf.geometry[region_index]
        if region_geometry.intersects(hotspot_polygons):
            filter_by_hotspot.append(region_id)

    region_gdf = region_gdf[region_gdf["region_code"].isin(filter_by_hotspot)].reindex()

    logger.info(
        "The number of region changes to %s  after Hot Spot filter",
        str(len(region_gdf)),
    )

    local_json_file = f"{task_id}-regions.json"

    region_gdf.to_file(local_json_file, driver="GeoJSON")

    s3 = boto3.client("s3")

    with open(local_json_file, "rb") as f:
        s3.upload_fileobj(
            f,
            BUCKET_NAME,
            f"{output_s3_folder}/{local_json_file}",
        )


@main.command(no_args_is_help=True)
@click.option(
    "--task-id",
    "-t",
    type=str,
    default=None,
    help="REQUIRED. Burn Cube task id, e.g. Dec-21.",
)
@click.option(
    "--output-s3-folder",
    "-o",
    type=str,
    default="projects/burn_cube/airflow-run/burn-cube-app/ancillary_file",
    help="The ancillary_file folder which save clean-up Hotspot CSV file.",
)
@click.option(
    "--task-table",
    "-b",
    type=str,
    default="10-year-historical-processing-4year-geomad.csv",
    help="The task table in configs folder, e.g. 10-year-historical-processing-4year-geomad.csv.",
)
@click.option("-v", "--verbose", count=True)
def update_hotspot_data(
    task_id,
    output_s3_folder,
    task_table,
    verbose,
):
    logging_setup(verbose)

    # use task_id to get the mappingperiod information to filter hotspot
    bc_running_task = utils.generate_task(task_id, task_table)

    mappingperiod = (
        bc_running_task["Mapping Period Start"],
        bc_running_task["Mapping Period End"],
    )

    logger.info("Use mappingperiod: %s to filter hotspot file", str(mappingperiod))

    import numpy as np

    start = (
        np.datetime64(mappingperiod[0]).astype("datetime64[ns]") - np.datetime64(2, "M")
    ).astype("datetime64[ns]")
    stop = np.datetime64(mappingperiod[1])

    # the current (10/01/2023) zip file size is 430MB. It is safe to download it to local file system
    import shutil

    import requests

    hotspot_product_url = (
        "https://ga-sentinel.s3-ap-southeast-2.amazonaws.com/historic/all-data-csv.zip"
    )
    filename = "all-data-csv.zip"
    csv_filename = "hotspot_historic.csv"

    r = requests.get(hotspot_product_url, stream=True)
    r.raw.decode_content = True
    with open(filename, "wb") as f:
        shutil.copyfileobj(r.raw, f)

    # load the CSV file from zip file
    import zipfile

    import pandas as pd

    with zipfile.ZipFile(filename) as z:
        # TODO: find a way to check the actual CSV file from zip file
        with z.open(csv_filename) as f:

            # only load these 4 columns from hotspot to save RAM
            column_names = ["datetime", "sensor", "latitude", "longitude"]

            # read the hotspot data as Pandas.DataFrame
            hotspot_df = pd.read_csv(f, usecols=column_names, low_memory=False)

            dates = pd.to_datetime(
                hotspot_df.datetime.apply(lambda x: x.split("+")[0]).values
            )

            # filter the dataframe: just filter by period and sensor information
            index = np.where(
                (hotspot_df.sensor == "MODIS") & (dates >= start) & (dates <= stop)
            )[0]

            filtered_df = hotspot_df[hotspot_df.index.isin(index)]

            filtered_csv = f"{task_id}-{csv_filename}"

            # save the current task hotspot information to its CSV file, and upload to S3 later
            filtered_df.to_csv(filtered_csv, index=False)

            s3 = boto3.client("s3")

            with open(filtered_csv, "rb") as f:
                s3.upload_fileobj(
                    f,
                    BUCKET_NAME,
                    f"{output_s3_folder}/{filtered_csv}",
                )


@main.command(no_args_is_help=True)
@click.option(
    "--task-id",
    "-t",
    type=str,
    default=None,
    help="REQUIRED. Burn Cube task id, e.g. Dec-21.",
)
@click.option(
    "--region-id",
    "-r",
    type=str,
    default=None,
    help="REQUIRED. Region id AU-30 Grid.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    # Don't mandate existence since this might be s3://.
    help="REQUIRED. Path to the output directory.",
)
@click.option(
    "--split-count",
    "-s",
    type=int,
    default=2,
    help="The number of sub-region from a signle au-30 Grid region.",
)
@click.option(
    "--geomed-product-name",
    "-g",
    type=str,
    help="The 4-year period GeoMED product name, e.g. ga_ls8c_nbart_gm_4cyear_3 or ga_ls8c_nbart_gm_4fyear_3.",
)
@click.option(
    "--task-table",
    "-b",
    type=str,
    default="10-year-historical-processing-4year-geomad.csv",
    help="The task table in configs folder, e.g. 10-year-historical-processing-4year-geomad.csv.",
)
@click.option("-v", "--verbose", count=True)
def burn_cube_run(
    task_id,
    region_id,
    output,
    split_count,
    geomed_product_name,
    task_table,
    verbose,
):

    # save the beginning value to display whole region processing duration
    burn_cube_process_beginning = datetime.now()

    burn_cube_process_timer = burn_cube_process_beginning

    logging_setup(verbose)

    bc_running_task = utils.generate_task(task_id, task_table)

    # geomed_bands = ["red", "green", "blue", "nir", "swir1", "swir2"]
    geomed_bands = ["green", "nir", "swir2"]

    # ard_bands = [
    #     f"nbart_{band}" for band in ("red", "green", "blue", "nir", "swir_1", "swir_2")
    # ]

    ard_bands = [f"nbart_{band}" for band in ("green", "nir", "swir_2")]

    period = (bc_running_task["Period Start"], bc_running_task["Period End"])
    mappingperiod = (
        bc_running_task["Mapping Period Start"],
        bc_running_task["Mapping Period End"],
    )

    # The following variables passed by K8s Pod manifest
    hnrs_config = {
        "db_hostname": os.getenv("HNRS_DB_HOSTNAME"),
        "db_password": os.getenv("HNRS_DC_DB_PASSWORD"),
        "db_username": os.getenv("HNRS_DC_DB_USERNAME"),
        "db_port": 5432,
        "db_database": os.getenv("HNRS_DC_DB_DATABASE"),
    }

    odc_config = {
        "db_hostname": os.getenv("ODC_DB_HOSTNAME"),
        "db_password": os.getenv("ODC_DB_PASSWORD"),
        "db_username": os.getenv("ODC_DB_USERNAME"),
        "db_port": 5432,
        "db_database": os.getenv("ODC_DB_DATABASE"),
    }

    logger.info("Use period: %s", str(period))

    logger.info("Use mappingperiod: %s", str(mappingperiod))

    dc = datacube.Datacube(app="Burn Cube K8s processing", config=odc_config)

    gpgon, geomed = get_geomed_ds(
        region_id, period, hnrs_config, geomed_bands, geomed_product_name
    )

    for x_i in range(split_count):
        for y_i in range(split_count):

            burn_cube_result, burn_cube_process_timer = generate_subregion_result(
                dc,
                geomed,
                ard_bands,
                period,
                mappingperiod,
                gpgon,
                task_id,
                region_id,
                x_i,
                y_i,
                split_count,
                burn_cube_process_timer,
                output,
            )

            if split_count != 1:
                s3_file_path = f"{task_id}/{region_id}/BurnMapping-{task_id}-{region_id}-{x_i}-{y_i}.nc"
                local_file_path = (
                    f"/tmp/BurnMapping-{task_id}-{region_id}-{x_i}-{y_i}.nc"
                )
            else:
                s3_file_path = (
                    f"{task_id}/{region_id}/BurnMapping-{task_id}-{region_id}.nc"
                )
                local_file_path = f"/tmp/BurnMapping-{task_id}-{region_id}.nc"

            from urllib.parse import urlparse

            o = urlparse(output)

            target_file_path = f"{o.path}/{s3_file_path}"

            # let us assume the output is an AWS S3 path
            if burn_cube_result:
                burn_cube_result_apply_wofs = apply_post_processing_by_wo_summary(
                    dc,
                    burn_cube_result,
                    gpgon,
                    mappingperiod,
                    x_i,
                    y_i,
                    split_count,
                    region_id,
                )

                burn_cube_process_timer = display_current_step_processing_duration(
                    log_text="The burn cube processing by wofs_summary",
                    burn_cube_process_timer=burn_cube_process_timer,
                    log_code=f"{task_id}-{region_id}-{x_i}-{y_i}",
                )

                comp = dict(zlib=True, complevel=5)
                encoding = {
                    var: comp for var in burn_cube_result_apply_wofs.data_vars
                }  # compression

                # this will save it in the current working directory
                burn_cube_result_apply_wofs.to_netcdf(
                    local_file_path, encoding=encoding
                )

                s3 = boto3.client("s3")

                with open(local_file_path, "rb") as f:
                    s3.upload_fileobj(f, o.netloc, target_file_path[1:])

                    logger.info("Upload NetCDF file to: %s", target_file_path)

                # use to_cog feature to convert each band from XArray.Dataset to COG
                for band, dv in burn_cube_result_apply_wofs.data_vars.items():
                    ds_output = burn_cube_result_apply_wofs[band].to_dataset(name=band)
                    ds_output.attrs["crs"] = geometry.CRS("EPSG:3577")
                    da_output = ds_output.to_array()

                    local_tiff_file = local_file_path.replace(
                        ".nc", f"-{band.lower()}.tif"
                    )

                    write_cog(geo_im=da_output, fname=local_tiff_file)

                    with open(local_tiff_file, "rb") as f:
                        s3.upload_fileobj(
                            f,
                            o.netloc,
                            target_file_path[1:].replace(".nc", f"-{band.lower()}.tif"),
                        )

                        logger.info(
                            "Upload GeoTiff file: %s",
                            target_file_path.replace(".nc", f"-{band.lower()}.tif"),
                        )

                burn_cube_process_timer = display_current_step_processing_duration(
                    log_text="The burn cube uploading result duration",
                    burn_cube_process_timer=burn_cube_process_timer,
                    log_code=f"{task_id}-{region_id}-{x_i}-{y_i}",
                )

    _ = display_current_step_processing_duration(
        log_text="The burn cube processing duration",
        burn_cube_process_timer=burn_cube_process_beginning,
        log_code=f"{task_id}-{region_id}",
    )


if __name__ == "__main__":
    main()
