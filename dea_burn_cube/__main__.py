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
import xarray as xr

import dea_burn_cube.__version__
import dea_burn_cube.utils as utils

logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def gqa_predicate(ds):
    return ds.metadata.gqa_iterative_mean_xy <= 1


def display_current_step_processing_duration(
    log_text, burn_cube_process_timer, log_code
):
    # display the datacube loading time
    now = datetime.now()
    duration = now - burn_cube_process_timer
    logger.info(f"{log_code}: {log_text} - {duration.total_seconds()} seconds")

    # remember to reset the timer
    return now


def get_geomed_ds(region_id, period, hnrs_config, geomed_bands):
    hnrs_dc = datacube.Datacube(app="geomed_loading", config=hnrs_config)

    metadata_files = hnrs_dc.find_datasets(product="ga_ls8c_nbart_gm_4cyear_3")

    # TODO: we should use au-30 grid to replace this section
    sample_dataset = [
        e
        for e in metadata_files
        if e.metadata_doc["properties"]["odc:region_code"] == region_id
    ][0]

    metadata = hnrs_dc.index.datasets.get(str(sample_dataset.id))

    geometry_list = [metadata.extent]

    region_polygon = gpd.GeoDataFrame(
        index=range(len(geometry_list)), crs="epsg:3577", geometry=geometry_list
    )

    gpgon = datacube.utils.geometry.Geometry(
        region_polygon.geometry[0], crs="epsg:3577"
    )

    # TODO: check the 4-year period will grab only one GeoMAD as we wish, or more than one
    geomed = hnrs_dc.load(
        "ga_ls8c_nbart_gm_4cyear_3",
        time=period,
        geopolygon=gpgon,
        resampling="nearest",
        group_by="solar_day",
        output_crs="EPSG:3577",
        resolution=(-30, 30),
        dask_chunks={},
    )

    geomed = geomed[geomed_bands].to_array(dim="band").to_dataset(name="geomedian")

    return gpgon, geomed


def apply_post_processing_by_wo_summary(
    dc, burn_cube_result, gpgon, mappingperiod, x_i, y_i, split_count
):

    # TODO: we dont use Dask to walkaround the loading issue here cause the WOfS result size is small
    wofs_summary = dc.load(
        "ga_ls_wo_fq_fyear_3",
        time=mappingperiod,
        geopolygon=gpgon,
        resampling="nearest",
        group_by="solar_day",
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

    return xr.Dataset(
        {
            "StartDate": burn_cube_result["StartDate"],
            "Duration": burn_cube_result["Duration"],
            "Severity": burn_cube_result["Severity"],
            "Severe": burn_cube_result["Severe"],
            "Moderate": burn_cube_result["Moderate"],
            "Corroborate": burn_cube_result["Corroborate"],
            "Cleaned": burn_cube_result["Cleaned"],
            "WOfSModerate": wofs_moderate,
            "WOfSSevere": wofs_severe,
            "WOfSSeverity": wofs_severity,
            "WOfSStartDate": wofs_startdate,
            "WOfSDuration": wofs_duration,
            "WOfSCorroborate": wofs_corroborate,
            "WOfSCleaned": wofs_cleaned,
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
):

    ard = dea_tools.datahandling.load_ard(
        dc,
        products=["ga_ls8c_ard_3"],
        measurements=ard_bands,
        geopolygon=gpgon,
        output_crs="EPSG:3577",
        resolution=(-30, 30),
        resampling={"fmask": "nearest", "*": "bilinear"},
        dask_chunks={},
        predicate=gqa_predicate,
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
        dask_chunks={},
        predicate=gqa_predicate,
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

    severitymapping_result = utils.severitymapping(
        mapping_dis, outliers_result, mappingperiod, method="NBR", growing=True
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
        if not name.startswith("fiona")
        and not name.startswith("sqlalchemy")
        and not name.startswith("boto")
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
    "--split_count",
    "-s",
    type=int,
    default=2,
    help="The number of sub-region from a signle au-30 Grid region.",
)
@click.option("-v", "--verbose", count=True)
def burn_cube_run(
    task_id,
    region_id,
    output,
    split_count,
    verbose,
):

    # save the beginning value to display whole region processing duration
    burn_cube_process_beginning = datetime.now()

    burn_cube_process_timer = burn_cube_process_beginning

    logging_setup(verbose)

    bc_running_task = utils.generate_task(task_id)

    geomed_bands = ["red", "green", "blue", "nir", "swir1", "swir2"]

    ard_bands = [
        f"nbart_{band}" for band in ("red", "green", "blue", "nir", "swir_1", "swir_2")
    ]

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

    logger.info(
        f"period:{period}, mappingperiod:{mappingperiod}, region:{region_id}, task_id:{task_id}"
    )

    dc = datacube.Datacube(app="Burn Cube K8s processing", config=odc_config)

    gpgon, geomed = get_geomed_ds(region_id, period, hnrs_config, geomed_bands)

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
            )

            s3_file_path = f"{task_id}/{region_id}/BurnMapping-{task_id}-{region_id}-{x_i}-{y_i}.nc"
            local_file_path = f"BurnMapping-{task_id}-{region_id}-{x_i}-{y_i}.nc"

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
