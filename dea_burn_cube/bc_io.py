"""
This module contains functions for input/output operations related to burn mapping using
the DEA Burn Cube and AWS S3.

"""


import logging

import boto3
import xarray as xr
from datacube.utils import geometry
from datacube.utils.cog import write_cog

from dea_burn_cube import helper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def upload_object_to_s3(local_file_name: str, s3_uri: str) -> None:
    """
    Uploads a local file to an S3 bucket.

    Args:
        local_file_name: The path of the local file to upload.
        s3_uri: The S3 URI where the file will be uploaded.

    Returns:
        None.
    """

    s3 = boto3.client("s3")

    s3_bucket_name, s3_object_key = helper.extract_s3_details(s3_uri)

    with open(local_file_name, "rb") as f:
        s3.upload_fileobj(
            f,
            s3_bucket_name,
            s3_object_key,
        )


def result_file_saving_and_uploading(
    burn_cube_result_apply_wofs: xr.Dataset,
    object_title: str,
    object_key: str,
    s3_bucket_name: str,
) -> None:
    """
    Saves burn cube result as netCDF file, converts each band to a GeoTIFF and uploads the files to an S3 bucket.

    Args:
        burn_cube_result_apply_wofs: An XArray Dataset object representing the result of the burn cube analysis.
        object_title: A string representing the path to save the local netCDF file.
        object_key: A string representing the target S3 bucket and prefix where the files will be uploaded.
        s3_bucket_name: A string representing the name of the S3 bucket.

    Returns:
        None.

    Raises:
        IOError: If there was an error reading or writing the files.

    Example:
        >>> result_file_saving_and_uploading(burn_cube_result,
                                            'burn_cube_result',
                                            's3://my-bucket/output',
                                            'my-bucket')
    """

    s3 = boto3.client("s3")

    # use to_cog feature to convert each band from XArray.Dataset to COG
    for band, _ in burn_cube_result_apply_wofs.data_vars.items():
        ds_output = burn_cube_result_apply_wofs[band].to_dataset(name=band)
        ds_output.attrs["crs"] = geometry.CRS("EPSG:3577")

        da_output = ds_output.to_array()

        local_tiff_file = object_title + f"-{band.lower()}.tif"

        write_cog(geo_im=da_output, fname=local_tiff_file, overwrite=True)

        with open(local_tiff_file, "rb") as f:
            s3.upload_fileobj(
                f,
                s3_bucket_name,
                object_key + f"-{band.lower()}.tif",
            )

            logger.info(
                "Upload GeoTiff file: %s",
                object_key + f"-{band.lower()}.tif",
            )
