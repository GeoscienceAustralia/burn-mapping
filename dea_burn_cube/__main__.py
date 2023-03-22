"""CLI: Run Burn Cube analysis based on task_id and region_id
Geoscience Australia
2022
"""

import logging
import os
import shutil
import sys
import zipfile
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
import xarray as xr
from datacube.utils import geometry
from datacube.utils.cog import write_cog
from shapely.geometry import Point
from shapely.ops import unary_union

import dea_burn_cube.__version__
from dea_burn_cube import bc_data_loading, bc_data_processing, io, task

logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

os.environ["SQLALCHEMY_SILENCE_UBER_WARNING"] = "1"


@task.log_execution_time
def result_file_saving_and_uploading(
    burn_cube_result_apply_wofs: xr.Dataset,
    local_file_path: str,
    object_key: str,
    bucket_name: str,
) -> None:
    """
    Saves burn cube result as netCDF file, converts each band to a GeoTIFF and uploads the files to an S3 bucket.

    Args:
        burn_cube_result_apply_wofs: An XArray Dataset object representing the result of the burn cube analysis.
        local_file_path: A string representing the path to save the local netCDF file.
        object_key: A string representing the target S3 bucket and prefix where the files will be uploaded.
        bucket_name: A string representing the name of the S3 bucket.

    Returns:
        None.

    Raises:
        IOError: If there was an error reading or writing the files.

    Example:
        >>> result_file_saving_and_uploading(burn_cube_result,
                                            '/tmp/burn_cube_result.nc',
                                            's3://my-bucket/output',
                                            'my-bucket')
    """

    comp = dict(zlib=True, complevel=5)
    encoding = {
        var: comp for var in burn_cube_result_apply_wofs.data_vars
    }  # compression

    # this will save it in the current working directory
    burn_cube_result_apply_wofs.to_netcdf(local_file_path, encoding=encoding, mode="w")

    s3 = boto3.client("s3")

    with open(local_file_path, "rb") as f:
        s3.upload_fileobj(f, bucket_name, object_key)

    # use to_cog feature to convert each band from XArray.Dataset to COG
    for band, _ in burn_cube_result_apply_wofs.data_vars.items():
        ds_output = burn_cube_result_apply_wofs[band].to_dataset(name=band)
        ds_output.attrs["crs"] = geometry.CRS("EPSG:3577")

        da_output = ds_output.to_array()

        local_tiff_file = local_file_path.replace(".nc", f"-{band.lower()}.tif")

        write_cog(geo_im=da_output, fname=local_tiff_file, overwrite=True)

        with open(local_tiff_file, "rb") as f:
            s3.upload_fileobj(
                f,
                bucket_name,
                object_key.replace(".nc", f"-{band.lower()}.tif"),
            )

            logger.info(
                "Upload GeoTiff file: %s",
                object_key.replace(".nc", f"-{band.lower()}.tif"),
            )


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
    "--region-list-s3-path",
    "-r",
    type=str,
    default=None,
    help="REQUIRED. The AU-30 Region list filter by Ocean Mask and Hotspot in GeoJSON format.",
)
@click.option(
    "--process-cfg-url",
    "-p",
    type=str,
    default=None,
    help="REQUIRED. The Path URL to Burn Cube process cfg file as YAML format.",
)
def filter_regions_by_output(task_id, region_list_s3_path, process_cfg_url):
    """
    There are one assumption about this method:
        1. we already run filter_regions method to get the region list

    """
    _ = s3fs.S3FileSystem(anon=True)

    _ = "s3" in gpd.io.file._VALID_URLS
    gpd.io.file._VALID_URLS.discard("s3")

    region_gdf = gpd.read_file(region_list_s3_path)
    region_gdf = region_gdf.to_crs(epsg="3577")

    logger.info("Filter %s by output NetCDF files", region_list_s3_path)

    process_cfg = task.load_yaml_remote(process_cfg_url)
    output = process_cfg["output_folder"]
    overwrite = process_cfg["overwrite"]
    platform = process_cfg["input_products"]["platform"]

    ancillary_folder = f"{output}/ancillary_file"

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

            _, target_file_path = task.generate_output_filenames(
                output, task_id, region_id, platform
            )

            if not task.check_file_exists(s3_bucket_name, target_file_path[1:]):
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
def filter_regions(task_id, region_list_s3_path, process_cfg_url):
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

    if task.check_file_exists(o.netloc, f"{ancillary_key}/{local_json_file}"):
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
def update_hotspot_data(
    task_id,
    process_cfg_url,
):
    logging_setup()

    process_cfg = task.load_yaml_remote(process_cfg_url)

    task_table = process_cfg["task_table"]
    output = process_cfg["output_folder"] + "/ancillary_file"

    o = urlparse(output)

    output_s3_folder = o.path[1:]

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
    csv_filename = "hotspot_historic.csv"
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

            filtered_csv = f"{task_id}-{csv_filename}"

            # save the current task hotspot information to its CSV file, and upload to S3 later
            filtered_df.to_csv(filtered_csv, index=False)

            s3 = boto3.client("s3")

            with open(filtered_csv, "rb") as f:
                s3.upload_fileobj(
                    f,
                    o.netloc,
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
    "--process-cfg-url",
    "-p",
    type=str,
    default=None,
    help="REQUIRED. The Path URL to Burn Cube process cfg file as YAML format.",
)
@click.option(
    "--n-procs",
    "-n",
    type=int,
    default=8,
    help="The size of process pool.",
)
def burn_cube_run(
    task_id,
    region_id,
    process_cfg_url,
    n_procs,
):
    """
    The main method to run Burn Cube processing based on task id and region ID.
    """

    logging_setup()

    process_cfg = task.load_yaml_remote(process_cfg_url)

    task_table = process_cfg["task_table"]
    geomed_product_name = process_cfg["input_products"]["geomed"]
    wofs_summary_product_name = process_cfg["input_products"]["wofs_summary"]
    ard_product_names = process_cfg["input_products"]["ard_product_names"]
    ard_bands = process_cfg["input_products"]["input_ard_bands"]
    geomed_bands = process_cfg["input_products"]["input_gm_bands"]
    platform = process_cfg["input_products"]["platform"]

    output = process_cfg["output_folder"]
    overwrite = process_cfg["overwrite"]

    bc_running_task = task.generate_task(task_id, task_table)

    # geomed_bands = ["red", "green", "blue", "nir", "swir1", "swir2"]  # 7 bands setting
    # geomed_bands = ["green", "nir", "swir2"] # 3 bands setting

    # 7 bands setting
    # ard_bands = [
    #    f"nbart_{band}" for band in ("red", "green", "blue", "nir", "swir_1", "swir_2")
    # ]

    # 3 bands setting
    # ard_bands = [f"nbart_{band}" for band in ("green", "nir", "swir_2")]

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

    odc_dc = datacube.Datacube(
        app=f"Burn Cube K8s processing - {region_id}", config=odc_config
    )
    hnrs_dc = datacube.Datacube(
        app=f"Burn Cube K8s processing - {region_id}", config=hnrs_config
    )

    # TODO: only check the NetCDF is not enough

    local_file_path, target_file_path = task.generate_output_filenames(
        output, task_id, region_id, platform
    )

    o = urlparse(output)

    bucket_name = o.netloc
    object_key = target_file_path[1:]

    if task.check_file_exists(bucket_name, object_key) and not overwrite:
        logger.info("Find NetCDF file %s in s3, skip it.", target_file_path)
    else:
        # check the input product detail
        # TODO: can add dry-run, and it will stop after input dataset list check
        try:
            gpgon, input_dataset_list = bc_data_loading.check_input_datasets(
                hnrs_dc,
                odc_dc,
                period,
                mappingperiod,
                geomed_product_name,
                wofs_summary_product_name,
                ard_product_names,
                region_id,
            )
        except bc_data_loading.IncorrectInputDataError:
            logger.error(
                "The input datasets have problem. finish the processing %s", region_id
            )
            # Not enough data to finish the processing, so stop it here
            sys.exit(0)

        logger.info("Will save NetCDF file as temp file to: %s", local_file_path)
        logger.info("Will upload NetCDF file to: %s", target_file_path)

        # After the check_input_datasets pass, we can use input information to generate
        # processing log
        processing_log = task.generate_processing_log(
            task_id,
            period,
            mappingperiod,
            geomed_product_name,
            wofs_summary_product_name,
            ard_product_names,
            region_id,
            output,
            task_table,
            input_dataset_list,
        )

        # No matter upload successful or not, should not block the main processing
        io.upload_dict_to_s3(
            processing_log, bucket_name, object_key.replace(".nc", ".json")
        )

        burn_cube_result = bc_data_processing.generate_bc_result(
            odc_dc,
            hnrs_dc,
            ard_product_names,
            geomed_product_name,
            ard_bands,
            geomed_bands,
            period,
            mappingperiod,
            gpgon,
            task_id,
            output,
            n_procs,
        )

        if burn_cube_result:
            burn_cube_result_apply_wofs = (
                bc_data_processing.apply_post_processing_by_wo_summary(
                    odc_dc,
                    burn_cube_result,
                    gpgon,
                    mappingperiod,
                    wofs_summary_product_name,
                )
            )

            # TODO: should use Try-Catch to know IO is OK or not
            result_file_saving_and_uploading(
                burn_cube_result_apply_wofs,
                local_file_path,
                object_key,
                bucket_name,
            )


if __name__ == "__main__":
    main()
