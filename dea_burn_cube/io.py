"""
This module contains functions for input/output operations related to burn mapping using
the DEA Burn Cube and AWS S3.

"""


import json
import logging
from typing import Dict

import boto3
import xarray as xr
from datacube.utils import geometry
from datacube.utils.cog import write_cog

from dea_burn_cube import helper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def upload_dict_to_s3(dictionary: Dict, bucket_name: str, file_name: str):
    """
    Uploads a Python dictionary as a JSON file to AWS S3.

    Args:
        dictionary: A Python dictionary to upload as a JSON file.
        bucket_name: The name of the S3 bucket to upload the JSON file to.
        file_name: The name to use for the uploaded JSON file in S3.

    """
    # Initialize the S3 resource and convert the dictionary to a JSON string
    s3 = boto3.resource("s3")
    json_string = json.dumps(dictionary, indent=4)

    # Upload the JSON string to S3
    try:
        s3.Object(bucket_name, file_name).put(Body=json_string)
    except Exception:
        logger.warning("Cannot upload the file to: %s", file_name)


def upload_object_to_s3(local_file_path: str, s3_uri: str) -> None:
    """
    Uploads a local file to an S3 bucket.

    Args:
        local_file_path: The path of the local file to upload.
        s3_uri: The S3 URI where the file will be uploaded.

    Returns:
        None.
    """

    s3 = boto3.client("s3")

    s3_bucket_name, s3_object_key = helper.extract_s3_details(s3_uri)

    with open(local_file_path, "rb") as f:
        s3.upload_fileobj(
            f,
            s3_bucket_name,
            s3_object_key,
        )


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
                                            'burn_cube_result.nc',
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
