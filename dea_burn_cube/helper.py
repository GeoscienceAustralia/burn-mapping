import datetime
import logging
import os
import time
from typing import Any, Dict, Tuple
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


def get_and_set_aws_credentials() -> Dict[str, str]:
    """
    Fetches the current AWS credentials based on the IAM role attached to the environment and sets them as
    environment variables.

    Returns:
        Dict[str, str]: A dictionary containing 'access_key', 'secret_key', and 'token'.
    """
    # Create a session using the default configuration (implicitly using the IAM role)
    session = boto3.Session()

    # Access the credentials
    credentials = session.get_credentials()

    # Get current, refreshed credentials
    current_credentials = credentials.get_frozen_credentials()

    # Set credentials as environment variables
    os.environ["AWS_ACCESS_KEY_ID"] = current_credentials.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = current_credentials.secret_key
    os.environ["AWS_SESSION_TOKEN"] = current_credentials.token

    # Prepare the credentials dictionary
    credentials_dict = {
        "access_key": os.environ["AWS_ACCESS_KEY_ID"],
        "secret_key": os.environ["AWS_SECRET_ACCESS_KEY"],
        "token": os.environ["AWS_SESSION_TOKEN"],
    }

    return credentials_dict


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


def extract_s3_details(uri: str) -> Tuple[str, str, str]:
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

    s3_bucket_name, s3_object_key = extract_s3_details(s3_file_uri)

    if s3_bucket_name is None or s3_object_key is None:
        error_message = f"Illegal S3 URI: {s3_file_uri}"
        logger.error(error_message)
        return False

    s3 = boto3.client("s3")

    try:
        s3.head_object(Bucket=s3_bucket_name, Key=s3_object_key)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        else:
            raise
    else:
        return True
