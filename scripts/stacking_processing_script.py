import logging
import sys

import click
import rioxarray
import s3fs
import xarray as xr
from datacube.utils import geometry
from datacube.utils.cog import write_cog

from dea_burn_cube import bc_io, helper, task

# Set up logging configurations
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def logging_setup():
    """
    Set up the logging configuration to display logs from all modules except certain libraries.
    """
    # Collect all loggers except those from specific libraries
    loggers = [
        logging.getLogger(name)
        for name in logging.root.manager.loggerDict
        if not name.startswith("sqlalchemy") and not name.startswith("boto")
    ]

    # Set up a stream handler to direct logs to stdout
    stdout_hdlr = logging.StreamHandler(sys.stdout)
    for logger in loggers:
        logger.addHandler(stdout_hdlr)
        logger.propagate = False  # Prevent logs from propagating to the root logger


def process_files(match_products, region_id, output_folder, condition):
    """
    Processes the files for the given products and region ID by fetching data from S3,
    applying product weights, and combining the results.

    Parameters:
    - match_products (list): List of product information including name, weight, and file extension.
    - region_id (str): The ID of the region to process.
    - output_folder (str): The base folder path where output files are stored.
    - condition (str): The way to convert sub-indicators result to single binary result.

    Returns:
    - xarray.DataArray or None: Returns a combined summary of the processed files or None if no files are found.
    """
    pair_files = []

    # Initialize the S3 filesystem with anonymous access
    fs = s3fs.S3FileSystem(anon=True)

    # Collect matching files for each product based on region and file extension
    for match_product in match_products:
        # Build the target folder path dynamically for each product
        target_folder = f"{output_folder}/{match_product['product_name']}/3-0-0/{region_id[:3]}/{region_id[3:]}/"

        logger.info(f"Try to query folder: {target_folder}")

        # List files in the target folder using S3 file system
        all_files = fs.glob(target_folder + "**")

        # Filter the list to only include files with the specified extension
        matching_files = [
            file for file in all_files if file.endswith(match_product["extension_name"])
        ]

        # If matching files are found, store the first file and its product weight
        if matching_files:
            pair_files.append(
                {
                    "file_path": matching_files[0],
                    "product_weight": match_product["product_weight"],
                }
            )

    # If no files matched the criteria, return None to indicate no further processing is needed
    if not pair_files:
        logger.info(f"cannot find any match file.")
        sys.exit("Cannot find any files from product folders")

    # Open and process all matching files, applying their respective weights
    da_list = []
    for pair_file in pair_files:
        # Open raster data from S3 using rioxarray
        da = rioxarray.open_rasterio(f"s3://{pair_file['file_path']}")
        # Multiply the data array by the product weight
        da_list.append(da * pair_file["product_weight"])

    # Combine the weighted data arrays along a new dimension and sum them to get the summary
    combined = xr.concat(da_list, dim="variable")
    # sum_summary = combined.sum(dim="variable")

    # Compute a binary mask based on the chosen condition
    if condition == "any":
        # Set pixel to 1 if any of the images have a non-zero pixel
        binary_mask = (combined > 0).any(dim="variable").astype("int32")
    elif condition == "majority":
        # Set pixel to 1 if the majority of images have a non-zero pixel
        threshold = combined.sizes["variable"] // 2  # majority threshold
        binary_mask = (combined > 0).sum(dim="variable") > threshold
        binary_mask = binary_mask.astype("int32")
    elif condition == "all":
        # Set pixel to 1 only if all images have a non-zero pixel
        binary_mask = (combined > 0).all(dim="variable").astype("int32")
    else:
        raise ValueError("Invalid condition. Choose from 'any', 'majority', or 'all'.")

    # Add the Coordinate Reference System (CRS) attribute to the output data array
    binary_mask.attrs["crs"] = geometry.CRS("EPSG:3577")

    return binary_mask


@click.command(no_args_is_help=True)
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
    required=True,
    help="REQUIRED. Region ID for the processing, e.g., AU-30 Grid.",
)
@click.option(
    "--process-cfg-url",
    "-p",
    type=str,
    required=True,
    help="REQUIRED. URL to the Stacking process configuration file (YAML format).",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Whether to rerun scenes that have already been processed.",
)
def stacking_processing(task_id, region_id, process_cfg_url, overwrite):
    """
    Load and process satellite imagery data to generate a stacking result saved as a GeoTIFF file.

    Parameters:
    - task_id (str): The unique identifier of the task.
    - region_id (str): Region ID to identify the area of interest.
    - process_cfg_url (str): URL of the YAML configuration file for process settings.
    - overwrite (bool): Flag to determine whether to overwrite existing files.
    """
    logging_setup()  # Initialize the logging setup

    # Load the process configuration from the provided YAML URL
    process_cfg = helper.load_yaml_remote(process_cfg_url)

    match_products = process_cfg["match_products"]

    processing_task: task.BurnCubeProcessingTask = (
        task.BurnCubeProcessingTask.from_config(
            cfg_url=process_cfg_url, task_id=task_id, region_id=region_id
        )
    )

    processing_task.validate_cfg()
    processing_task.validate_data()

    # generate all kinds of conditions to do result comparision
    conditions = ["any", "majority", "all"]  # Options: "any", "majority", "all"

    for condition in conditions:
        # Process files based on the region and products information
        sum_summary = process_files(
            match_products, region_id, processing_task.output_folder, condition
        )

        # Define the output GeoTIFF file name pattern
        pred_tif = f"{condition}.tif"

        # Write the result to a Cloud Optimized GeoTIFF (COG) file
        write_cog(geo_im=sum_summary, fname=pred_tif, overwrite=overwrite, nodata=-999)

        logger.info(f"Saved result as: {pred_tif}")

        # Construct the S3 file URI for the output file
        s3_file_uri = f"{processing_task.s3_bucket_name}/{processing_task.s3_object_key}_{pred_tif}"

        # Activate AWS credentials from the service account attached
        helper.get_and_set_aws_credentials()

        # Upload the output GeoTIFF to the specified S3 location
        bc_io.upload_object_to_s3(pred_tif, s3_file_uri)
        logger.info(f"Uploaded to S3: {s3_file_uri}")

    # processing_task.s3_object_key
    processing_task.upload_processing_log()
    processing_task.add_metadata_files()


if __name__ == "__main__":
    stacking_processing()
