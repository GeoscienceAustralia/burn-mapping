"""CLI: Run Burn Cube analysis based on task_id and region_id
Geoscience Australia
2022
"""

import logging
import os
import sys
from multiprocessing import cpu_count

import click

import dea_burn_cube.__version__
from dea_burn_cube import bc_data_processing, helper, io, task

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

    logging_setup()

    bc_filter_task = task.BurnCubeFilterTask(process_cfg_url, task_id)

    if not overwrite and helper.check_s3_file_exists(
        bc_filter_task.region_list_s3_uri.replace(
            "-regions.json", "-cleanup-regions.json"
        )
    ):
        sys.exit(0)

    not_run_geojson = bc_filter_task.filter_by_output()

    not_run_geojson.to_file(
        bc_filter_task.region_list_local_uri.replace(
            "-regions.json", "-cleanup-regions.json"
        ),
        driver="GeoJSON",
    )

    io.upload_object_to_s3(
        bc_filter_task.region_list_local_uri.replace(
            "-regions.json", "-cleanup-regions.json"
        ),
        bc_filter_task.region_list_s3_uri.replace(
            "-regions.json", "-cleanup-regions.json"
        ),
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

    bc_filter_task = task.BurnCubeFilterTask(process_cfg_url, task_id)

    if not overwrite and helper.check_s3_file_exists(bc_filter_task.region_list_s3_uri):
        sys.exit(0)

    region_gdf = bc_filter_task.filter_by_region(region_list_s3_path)

    region_gdf.to_file(bc_filter_task.region_list_local_uri, driver="GeoJSON")

    io.upload_object_to_s3(
        bc_filter_task.region_list_local_uri, bc_filter_task.region_list_s3_uri
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

    bc_filter_task = task.BurnCubeFilterTask(process_cfg_url, task_id)

    # not need to regenerate the hotspot file because it
    # always same with the same task-id
    if not overwrite and helper.check_s3_file_exists(bc_filter_task.hotspot_csv_s3_uri):
        logger.info(
            "Find Hotspot file %s in s3, skip it.",
            bc_filter_task.hotspot_csv_s3_uri,
        )
        sys.exit(0)

    filtered_df = bc_filter_task.filter_by_hotspot()

    # save the current task hotspot information to its CSV file, and upload to S3 later
    filtered_df.to_csv(bc_filter_task.hotspot_csv_local_uri, index=False)

    io.upload_object_to_s3(
        bc_filter_task.hotspot_csv_local_uri, bc_filter_task.hotspot_csv_s3_uri
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

    bc_processing_task: task.BurnCubeProcessingTask = (
        task.BurnCubeProcessingTask.from_config(
            cfg_url=process_cfg_url, task_id=task_id, region_id=region_id
        )
    )

    if not overwrite and helper.check_s3_file_exists(
        bc_processing_task.stac_metadata_path
    ):
        logger.info(
            "Find metadata file %s in s3, skip it.",
            bc_processing_task.stac_metadata_path,
        )
        sys.exit(0)

    try:
        bc_processing_task.validate_cfg()
        bc_processing_task.validate_data()
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

    bc_processing_task.upload_processing_log()
    bc_processing_task.add_metadata()


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

    bc_processing_task: task.BurnCubeProcessingTask = (
        task.BurnCubeProcessingTask.from_config(
            cfg_url=process_cfg_url, task_id=task_id, region_id=region_id
        )
    )

    try:
        bc_processing_task.validate_cfg()
        bc_processing_task.validate_data()
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

    if not overwrite and helper.check_s3_file_exists(bc_processing_task.s3_file_uri):
        logger.info(
            "Find NetCDF file %s in s3, skip it.", bc_processing_task.s3_object_key
        )
        sys.exit(0)

    logger.info(
        "Will save NetCDF file as temp file to: %s", bc_processing_task.local_file_name
    )
    logger.info("Will upload NetCDF file to: %s", bc_processing_task.s3_file_uri)

    try:
        burn_cube_result = bc_data_processing.generate_bc_result(
            bc_processing_task,
            n_procs,
        )

        io.result_file_saving_and_uploading(
            burn_cube_result,
            bc_processing_task.local_file_name,
            bc_processing_task.s3_object_key,
            bc_processing_task.s3_bucket_name,
        )
    except Exception as e:
        logger.error(
            f"Generate and upload Burn Cube result to S3 object {bc_processing_task.s3_file_uri} failed: {str(e)}"
        )
        sys.exit(0)

    # then add metadata
    bc_processing_task.upload_processing_log()
    bc_processing_task.add_metadata()


if __name__ == "__main__":
    main()
