"""CLI: Run Burn Cube analysis based on task_id and region_id
Geoscience Australia
2022
"""

import logging
import os
import sys
from typing import List, Tuple
from urllib.parse import urlparse

import boto3
import botocore
import click
import datacube
import geopandas as gpd
import pandas as pd
import s3fs
import xarray as xr
from datacube.utils import geometry
from datacube.utils.cog import write_cog
from shapely.ops import unary_union

import dea_burn_cube.__version__
import dea_burn_cube.algo as algo
import dea_burn_cube.bc_data_loading as bc_data_loading
import dea_burn_cube.io as io
import dea_burn_cube.task as task

logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

BUCKET_NAME = "dea-public-data-dev"

os.environ["SQLALCHEMY_SILENCE_UBER_WARNING"] = "1"


def check_file_exists(bucket_name, file_key):
    """
    Checks if a file exists in an S3 bucket.

    :param bucket_name: The name of the S3 bucket.
    :param file_key: The key of the file in the bucket.
    :return: True if the file exists, False otherwise.
    """
    s3 = boto3.client("s3")

    try:
        s3.head_object(Bucket=bucket_name, Key=file_key)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        else:
            raise
    else:
        return True


def generate_output_filenames(output, task_id, region_id):
    s3_file_path = f"{task_id}/{region_id}/BurnMapping-{task_id}-{region_id}.nc"
    local_file_path = f"/tmp/BurnMapping-{task_id}-{region_id}.nc"

    o = urlparse(output)

    target_file_path = f"{o.path}/{s3_file_path}"

    return local_file_path, target_file_path


@task.log_execution_time
def result_file_saving_and_uploading(
    burn_cube_result_apply_wofs, local_file_path, target_file_path, bucket_name
):
    comp = dict(zlib=True, complevel=5)
    encoding = {
        var: comp for var in burn_cube_result_apply_wofs.data_vars
    }  # compression

    # this will save it in the current working directory
    burn_cube_result_apply_wofs.to_netcdf(local_file_path, encoding=encoding, mode="w")

    s3 = boto3.client("s3")

    with open(local_file_path, "rb") as f:
        s3.upload_fileobj(f, bucket_name, target_file_path[1:])

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
                target_file_path[1:].replace(".nc", f"-{band.lower()}.tif"),
            )

            logger.info(
                "Upload GeoTiff file: %s",
                target_file_path.replace(".nc", f"-{band.lower()}.tif"),
            )


@task.log_execution_time
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


@task.log_execution_time
def apply_post_processing_by_wo_summary(
    odc_dc, burn_cube_result, gpgon, mappingperiod, wofs_summary_product_name
):

    # TODO: we dont use Dask to walkaround the loading issue here cause the WOfS result size is small
    wofs_summary = bc_data_loading.load_wofs_summary_ds(
        odc_dc, gpgon, mappingperiod, wofs_summary_product_name
    )

    wofs_summary_frequency = wofs_summary.frequency

    wofs_summary_frequency = wofs_summary_frequency.load()

    wofs_mask = (wofs_summary_frequency[0, :, :].values < 0.2).astype(float)

    # ocean_mask = generate_ocean_mask(wofs_summary_frequency, region_id)

    burnpixel_mod = algo.burnpixel_masking(
        burn_cube_result, "Moderate"
    )  # mask the burnt area with "Medium" burnt area
    burnpixel_sev = algo.burnpixel_masking(burn_cube_result, "Severe")

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


@task.log_execution_time
def generate_reference_result(ard, geomed):
    dis = algo.distances(ard, geomed)
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
    )

    outliers_result = generate_reference_result(ard, geomed)

    del ard

    logger.info("Begin to load mapping data")
    mapping_ard = bc_data_loading.load_mapping_data(
        odc_dc,
        ard_product_names,
        ard_bands,
        mappingperiod,
        gpgon,
    )

    mapping_dis = algo.distances(mapping_ard, geomed)

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
    )

    return severitymapping_result


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
    bc_running_task = task.generate_task(task_id, task_table)

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
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Rerun region that have already been processed.",
)
@click.option("-v", "--verbose", count=True)
def burn_cube_run(
    task_id,
    region_id,
    output,
    split_count,
    geomed_product_name,
    task_table,
    overwrite,
    verbose,
):

    logging_setup(verbose)

    bc_running_task = task.generate_task(task_id, task_table)

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

    odc_dc = datacube.Datacube(
        app=f"Burn Cube K8s processing - {region_id}", config=odc_config
    )
    hnrs_dc = datacube.Datacube(
        app=f"Burn Cube K8s processing - {region_id}", config=hnrs_config
    )

    wofs_summary_product_name = "ga_ls_wo_fq_cyear_3"
    ard_product_names = ["ga_ls8c_ard_3"]

    # TODO: only check the NetCDF is not enough

    local_file_path, target_file_path = generate_output_filenames(
        output, task_id, region_id
    )

    o = urlparse(output)

    if check_file_exists(o.netloc, target_file_path[1:]) and not overwrite:
        logger.info("Find NetCDF file %s in s3, skip it.", target_file_path)
    else:
        # check the input product detail
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
        io.upload_processing_log(
            processing_log, o.netloc, target_file_path[1:].replace(".nc", ".json")
        )

        burn_cube_result = generate_bc_result(
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
        )

        if burn_cube_result:
            burn_cube_result_apply_wofs = apply_post_processing_by_wo_summary(
                odc_dc,
                burn_cube_result,
                gpgon,
                mappingperiod,
                wofs_summary_product_name,
            )

            # TODO: should use Try-Catch to know IO is OK or not
            result_file_saving_and_uploading(
                burn_cube_result_apply_wofs,
                local_file_path,
                target_file_path,
                o.netloc,
            )


if __name__ == "__main__":
    main()
