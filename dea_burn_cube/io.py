import json
import logging
from typing import Dict

import boto3

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
        logger.warning("Cannot upload the processing log file to: %s", file_name)
