"""CLI: Run Burn Cube analysis based on task_id and region_id
Geoscience Australia
2022
"""

import logging
import os
import shutil
import sys
import zipfile
from multiprocessing import cpu_count
from urllib.parse import urlparse

import boto3
import click
import datacube
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import requests
import s3fs
from shapely.geometry import Point
from shapely.ops import unary_union

import dea_burn_cube.__version__
from dea_burn_cube import bc_data_processing, io, task

logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

os.environ["SQLALCHEMY_SILENCE_UBER_WARNING"] = "1"


def logging_setup():
    """Set up logging."""
    loggers = [
        logging.getLogger(name)
        for name in logging.root.manager.loggerDict
        if not name.startswith("sqlalchemy") and not name.startswith("boto")
    ]

    stdout_hdlr = logging.StreamHandler(sys.stdout)
    for logger in loggers:
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
    "--process-cfg-url",
    "-p",
    type=str,
    default=None,
    help="REQUIRED. The Path URL to Burn Cube process cfg file as YAML format.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Rerun scenes that have already been processed.",
)
def filter_regions_by_output(task_id, process_cfg_url, overwrite):
    """
    There are one assumption about this method:
        1. we already run filter_regions method to get the region list

    """

    process_cfg = task.load_yaml_remote(process_cfg_url)
    output = process_cfg["output_folder"]
    platform = process_cfg["input_products"]["platform"]

    ancillary_folder = f"{output}/ancillary_file"

    _ = s3fs.S3FileSystem(anon=True)

    _ = "s3" in gpd.io.file._VALID_URLS
    gpd.io.file._VALID_URLS.discard("s3")

    region_list_s3_path = f"{ancillary_folder}/{task_id}-regions.json"

    region_gdf = gpd.read_file(region_list_s3_path)
    region_gdf = region_gdf.to_crs(epsg="3577")

    logger.info("Filter %s by output NetCDF files", region_list_s3_path)

    o = urlparse(ancillary_folder)
    s3_bucket_name = o.netloc
    ancillary_key = o.path[1:]

    local_json_file = f"{task_id}-cleanup-regions.json"

    # if we run it as overwrite mode, should not use NetCDF to filter region list
    if overwrite:
        not_run_geojson = region_gdf
    else:
        not_run_regions = []
        for region_index in region_gdf.index:
            region_id = region_gdf.region_code[region_index]

            _, s3_key_path, _ = task.generate_output_filenames(
                output, task_id, region_id, platform
            )

            if not task.check_file_exists(s3_bucket_name, s3_key_path):
                not_run_regions.append(region_id)

        not_run_geojson = region_gdf[
            region_gdf["region_code"].isin(not_run_regions)
        ].reindex()

    not_run_geojson.to_file(local_json_file, driver="GeoJSON")

    s3 = boto3.client("s3")

    with open(local_json_file, "rb") as f:
        s3.upload_fileobj(
            f,
            s3_bucket_name,
            f"{ancillary_key}/{local_json_file}",
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
    "--region-list-s3-path",
    "-r",
    type=str,
    default=None,
    help="REQUIRED. The AU-30 Region list in GeoJSON format.",
)
@click.option(
    "--process-cfg-url",
    "-p",
    type=str,
    default=None,
    help="REQUIRED. The Path URL to Burn Cube process cfg file as YAML format.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Rerun scenes that have already been processed.",
)
def filter_regions(task_id, region_list_s3_path, process_cfg_url, overwrite):
    """
    There are two assumptions on this method:
    1. user always use AU-30 grid standard GeoJSON
        The example region-list:
            s3://dea-public-data-dev/mangroves_aux/easter_vic.geojson
    2. user already updated hotspot, and we can use the clean-up CSV file
    """
    logging_setup()

    process_cfg = task.load_yaml_remote(process_cfg_url)

    ancillary_folder = process_cfg["output_folder"] + "/ancillary_file"

    o = urlparse(ancillary_folder)
    ancillary_key = o.path[1:]
    local_json_file = f"{task_id}-regions.json"

    # if we already had the region, skip it
    # TODO: must change the output name with region_list_s3_path

    if not overwrite and task.check_file_exists(
        o.netloc, f"{ancillary_key}/{local_json_file}"
    ):
        sys.exit(0)

    _ = s3fs.S3FileSystem(anon=True)

    _ = "s3" in gpd.io.file._VALID_URLS
    gpd.io.file._VALID_URLS.discard("s3")

    region_gdf = gpd.read_file(region_list_s3_path)
    region_gdf = region_gdf.to_crs(epsg="3577")

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

    # shuffle the region list to aviod data skew
    region_gdf = region_gdf.sample(frac=1).reset_index(drop=True)

    logger.info(
        "The number of region changes to %s  after Hot Spot filter",
        str(len(region_gdf)),
    )

    region_gdf.to_file(local_json_file, driver="GeoJSON")

    s3 = boto3.client("s3")

    with open(local_json_file, "rb") as f:
        s3.upload_fileobj(
            f,
            o.netloc,
            f"{ancillary_key}/{local_json_file}",
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
    "--process-cfg-url",
    "-p",
    type=str,
    default=None,
    help="REQUIRED. The Path URL to Burn Cube process cfg file as YAML format.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Rerun scenes that have already been processed.",
)
def update_hotspot_data(
    task_id,
    process_cfg_url,
    overwrite,
):
    logging_setup()

    process_cfg = task.load_yaml_remote(process_cfg_url)

    task_table = process_cfg["task_table"]
    output = process_cfg["output_folder"] + "/ancillary_file"

    o = urlparse(output)

    s3_bucket_name = o.netloc
    output_s3_folder = o.path[1:]

    csv_filename = "hotspot_historic.csv"

    filtered_csv = f"{task_id}-{csv_filename}"
    s3_file_uri = f"{output_s3_folder}/{filtered_csv}"

    # not need to regenerate the hotspot file because it
    # always same with the same task-id
    if not overwrite and task.check_file_exists(s3_bucket_name, s3_file_uri):
        sys.exit(0)

    # use task_id to get the mappingperiod information to filter hotspot
    bc_running_task = task.generate_task(task_id, task_table)

    mappingperiod = (
        bc_running_task["Mapping Period Start"],
        bc_running_task["Mapping Period End"],
    )

    logger.info("Use mappingperiod: %s to filter hotspot file", str(mappingperiod))

    start = (
        np.datetime64(mappingperiod[0]).astype("datetime64[ns]") - np.datetime64(2, "M")
    ).astype("datetime64[ns]")
    stop = np.datetime64(mappingperiod[1])

    # the current (10/01/2023) zip file size is 430MB. It is safe to download it to local file system

    hotspot_product_url = (
        "https://ga-sentinel.s3-ap-southeast-2.amazonaws.com/historic/all-data-csv.zip"
    )
    filename = "all-data-csv.zip"

    r = requests.get(hotspot_product_url, stream=True)
    r.raw.decode_content = True
    with open(filename, "wb") as f:
        shutil.copyfileobj(r.raw, f)

    # load the CSV file from zip file
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

            # save the current task hotspot information to its CSV file, and upload to S3 later
            filtered_df.to_csv(filtered_csv, index=False)

            s3 = boto3.client("s3")

            with open(filtered_csv, "rb") as f:
                s3.upload_fileobj(
                    f,
                    s3_bucket_name,
                    s3_file_uri,
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
    "--process-cfg-url",
    "-p",
    type=str,
    default=None,
    help="REQUIRED. The Path URL to Burn Cube process cfg file as YAML format.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Rerun scenes that have already been processed.",
)
def burn_cube_add_metadata(
    task_id,
    region_id,
    process_cfg_url,
    overwrite,
):
    logging_setup()

    task.add_metadata(task_id, region_id, process_cfg_url, overwrite)


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
    "--process-cfg-url",
    "-p",
    type=str,
    default=None,
    help="REQUIRED. The Path URL to Burn Cube process cfg file as YAML format.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Rerun scenes that have already been processed.",
)
def burn_cube_run(
    task_id,
    region_id,
    process_cfg_url,
    overwrite,
):
    """
    The main method to run Burn Cube processing based on task id and region ID.
    """

    n_procs = cpu_count()

    logging_setup()

    bc_task: task.BurnCubeProcessingTask = task.BurnCubeProcessingTask.from_config(
        cfg_url=process_cfg_url, task_id=task_id, region_id=region_id
    )

    try:
        bc_task.validate_cfg()
        bc_task.validate_data()
    except ValueError:
        logger.error(
            "The setting values in cfg have problem. finish the processing %s",
            region_id,
        )
        sys.exit(0)
    except task.IncorrectInputDataError:
        logger.error(
            "The input datasets have problem. finish the processing %s", region_id
        )
        # Not enough data to finish the processing, so stop it here
        sys.exit(0)

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

    logger.info("Use period: %s", str((bc_task.period_start, bc_task.period_end)))

    logger.info(
        "Use mappingperiod: %s",
        str((bc_task.mapping_period_start, bc_task.mapping_period_end)),
    )

    odc_dc = datacube.Datacube(
        app=f"Burn Cube K8s processing - {region_id}", config=odc_config
    )
    hnrs_dc = datacube.Datacube(
        app=f"Burn Cube K8s processing - {region_id}", config=hnrs_config
    )

    if not overwrite and task.check_file_exists(
        bc_task.bucket_name, bc_task.s3_key_path
    ):
        logger.info("Find NetCDF file %s in s3, skip it.", bc_task.s3_key_path)

        logger.info(
            "Will save NetCDF file as temp file to: %s", bc_task.local_file_path
        )
        logger.info("Will upload NetCDF file to: %s", bc_task.s3_file_path)

        burn_cube_result = bc_data_processing.generate_bc_result(
            odc_dc,
            hnrs_dc,
            bc_task.input_products.ard_product_names,
            bc_task.input_products.geomed,
            bc_task.input_products.input_ard_bands,
            bc_task.input_products.input_gm_bands,
            (bc_task.period_start, bc_task.period_end),
            (bc_task.mapping_period_start, bc_task.mapping_period_end),
            bc_task.gpgon,
            task_id,
            bc_task.output_folder,
            n_procs,
            bc_task.input_products.platform,
        )

        if burn_cube_result:
            burn_cube_result_apply_wofs = (
                bc_data_processing.apply_post_processing_by_wo_summary(
                    odc_dc,
                    burn_cube_result,
                    bc_task.gpgon,
                    (bc_task.mapping_period_start, bc_task.mapping_period_end),
                    bc_task.input_products.wofs_summary,
                )
            )

            # TODO: should use Try-Catch to know IO is OK or not
            io.result_file_saving_and_uploading(
                burn_cube_result_apply_wofs,
                bc_task.local_file_path,
                bc_task.s3_key_path,
                bc_task.bucket_name,
            )

            # then add metadata
            task.add_metadata(
                task_id,
                region_id,
                process_cfg_url,
                overwrite,
            )


if __name__ == "__main__":
    main()
