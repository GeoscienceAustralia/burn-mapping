import datetime
import logging
import time
from typing import Any, Dict
from urllib.parse import urlparse

import boto3
import botocore
import fsspec
import yaml
from datacube.utils.dates import normalise_dt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def log_execution_time(func):
    """
    Decorator function that logs the execution time of the decorated function.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function that logs the execution time.
    """

    def wrapper(*args, **kwargs):
        """
        Wrapper function that logs the execution time of the decorated function.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            object: Result of the decorated function.
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"{func.__name__} took {duration:.2f} seconds to execute.")
        return result

    return wrapper


def format_datetime(dt: datetime, with_tz=True, timespec="microseconds") -> str:
    dt = normalise_dt(dt)
    dt = dt.isoformat(timespec=timespec)
    if with_tz:
        dt = dt + "Z"
    return dt


def load_yaml_remote(yaml_url: str) -> Dict[str, Any]:
    """
    Open a yaml file remotely and return the parsed yaml document
    """
    try:
        with fsspec.open(yaml_url, mode="r") as f:
            return next(yaml.safe_load_all(f))
    except Exception:
        logger.error(f"Cannot load {yaml_url}")
        raise


def extract_s3_details(uri: str) -> tuple[str, str]:
    """
    Extracts the S3 bucket and object key from an S3 URI.

    Args:
        uri: The S3 URI.

    Returns:
        A tuple containing the bucket name and object key.
        If the URI is not in the S3 format, returns (None, None).
    """
    parsed_uri = urlparse(uri)

    if parsed_uri.scheme == "s3" and parsed_uri.netloc:
        bucket = parsed_uri.netloc
        object_key = parsed_uri.path.lstrip("/")
        return bucket, object_key
    else:
        error_message = f"Invalid S3 URI: {uri}"
        logger.error(error_message)
        return None, None


def check_s3_file_exists(s3_file_uri: str) -> bool:
    """
    Checks if a file exists in an S3 bucket.

    Args:
        s3_file_uri: The S3 URI of the file.

    Returns:
        True if the file exists, False otherwise.
    """

    bucket_name, file_key = extract_s3_details(s3_file_uri)

    if bucket_name is None or file_key is None:
        error_message = f"Illegal S3 URI: {s3_file_uri}"
        logger.error(error_message)
        return False

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
