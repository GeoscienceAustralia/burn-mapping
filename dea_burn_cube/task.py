"""
This module is used to create and manage tasks for burn mapping
using the DEA Burn Cube.

"""

import calendar
import datetime
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple
from urllib.parse import urlparse
from uuid import UUID, uuid5

import boto3
import botocore
import fsspec
import pandas as pd
import s3fs
import yaml

import dea_burn_cube.__version__ as version

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# Some random UUID to be ODC namespace
# copy paste from: odc-stats Model.py design
ODC_NS = UUID("6f34c6f4-13d6-43c0-8e4e-42b6c13203af")


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


def generate_output_filenames(
    output: str, task_id: str, region_id: str, platform: str
) -> Tuple[str, str]:
    """
    Generate local and target file paths for output file.

    Args:
        output: A string representing the S3 bucket and prefix where the output file will be stored.
        task_id: A string representing the ID of the task.
        region_id: A string representing the ID of the region.
        platform: A string representing the platform, e.g. s2 or ls.

    Returns:
        A tuple of strings representing the local file path and target file path.

    Example:
        >>> generate_output_filenames('s3://my-bucket/my-folder', '123', 'ABC')
        ('BurnMapping-123-ABC.nc', 's3://my-bucket/my-folder/123/ABC/BurnMapping-123-ABC.nc')
    """
    bc_output_file_path = (
        f"{task_id}/{region_id}/BurnMapping-{platform}-{task_id}-{region_id}.nc"
    )

    local_file_path = bc_output_file_path.split("/")[-1]

    o = urlparse(output)

    s3_bucket_name = o.netloc
    s3_folder_path = o.path[1:]

    s3_key_path = f"{s3_folder_path}/{bc_output_file_path}"

    return local_file_path, s3_key_path, s3_bucket_name


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


def task_to_ranges(task_id: str, task_table: str) -> Dict[str, str]:
    """
    Retrieves the start and end dates for the given task ID from the specified task table.

    Args:
        task_id : str
            The unique identifier of the task.
        task_table : str
            The name of the CSV file containing the task information.

    Returns:
        A dictionary containing the start and end dates for the processing period and mapping period,
        formatted as strings in the format "YYYY-MM-DD".
    """

    _ = s3fs.S3FileSystem(anon=True)

    periods_columns = [
        "Period Start",
        "Period End",
        "Mapping Period Start",
        "Mapping Period End",
    ]

    task_map = pd.read_csv(
        f"s3://dea-public-data-dev/projects/burn_cube/configs/{task_table}",
        parse_dates=periods_columns,
        dayfirst=True,
    )

    result_dict = {}

    if task_id not in list(task_map["Processing Name"]):
        return result_dict

    task_info = task_map[task_map["Processing Name"] == task_id].iloc[0]

    for periods_column in periods_columns:
        result_dict[periods_column] = task_info[periods_column].strftime("%Y-%m-%d")

    return result_dict


def dynamic_task_to_ranges(dtime: datetime.datetime) -> Dict[str, str]:
    """
    Generates a dictionary containing the start and end dates of the processing and mapping periods for
    a given datetime.

    Args:
        dtime: datetime.datetime
            The datetime to generate processing and mapping periods for.

    Returns:
        dict: A dictionary containing the start and end dates of the processing and mapping periods.
            The keys are: "Period Start", "Period End", "Mapping Period Start", and "Mapping Period End".
    """

    period_last_day = calendar.monthrange(dtime.year - 1, dtime.month)[1]

    result_dict = {}

    # Set the start date of the processing period to 5 years before the given datetime month + 1.
    # Set the end date of the processing period to the last day of the given datetime month of the previous year.
    result_dict["Period Start"] = datetime.datetime(
        dtime.year - 5, dtime.month + 1, 1
    ).strftime("%Y-%m-%d")
    result_dict["Period End"] = datetime.datetime(
        dtime.year - 1, dtime.month, period_last_day
    ).strftime("%Y-%m-%d")

    mapping_period_last_day = calendar.monthrange(dtime.year, dtime.month)[1]

    # Set the start date of the mapping period to the month + 1 of the previous year of the given datetime.
    # Set the end date of the mapping period to the last day of the given datetime month.
    result_dict["Mapping Period Start"] = datetime.datetime(
        dtime.year - 1, dtime.month + 1, 1
    ).strftime("%Y-%m-%d")
    result_dict["Mapping Period End"] = datetime.datetime(
        dtime.year, dtime.month, mapping_period_last_day
    ).strftime("%Y-%m-%d")

    return result_dict


def generate_task(task_id: str, task_table: str) -> Dict[str, str]:
    """
    Generate a dictionary of time ranges based on a given task ID and task table.

    Parameters
    ----------
    task_id : str
        The unique identifier of the task.
    task_table : str
        The name of the task table to read the time ranges from.

    Returns
    -------
    Dict[str, str]
        A dictionary containing the start and end times for the processing and mapping periods
        as strings formatted as "%Y-%m-%d".
    """
    # Parse the task ID to get the date
    updated_task_id = f"{task_id.split('-')[0]}-20{task_id.split('-')[1]}"
    dtime = datetime.datetime.strptime(updated_task_id, "%b-%Y")

    # Determine if the task falls within a dynamic or static time range and generate the
    # corresponding dictionary of time ranges
    if dtime.year > 2023:
        result_dict = dynamic_task_to_ranges(dtime)
    else:
        result_dict = task_to_ranges(task_id, task_table)

    return result_dict


def generate_processing_log(
    task_id: str,
    period: List[str],
    mappingperiod: List[str],
    geomed_product_name: str,
    wofs_summary_product_name: str,
    ard_product_names: List[str],
    region_id: str,
    output: str,
    task_table: str,
    summary_datasets: List[Dict[str, Any]],
    ard_datasets: List[str],
) -> Dict[str, Any]:
    """
    Generates a processing log dictionary for the task.

    Args:
    task_id : str
        ID of the task being processed
    period : List[str]
        Period to process data for
    mappingperiod : List[str])
        Mapping period to process data for
    geomed_product_name : str)
        Name of the GeoMAD product
    wofs_summary_product_name : str
        Name of the WOfS summary product
    ard_product_names : List[str]
        List of reference ARD products
    region_id : str
        Region ID to process data for
    output : str
        Output folder path
    task_table : str
        Name of the table to store the task
    summary_datasets : List[Dict[str, Any]]
        List of summary datasets with UUID and label
    ard_datasets: List[str]
        List of input ARD datasets with UUID only

    Returns:
    processing_log : Dict[str, Any]): A dictionary containing processing log information
    """

    # We can change it to EODatasets3 format in the future

    return {
        "task_id": task_id,
        "period": period,
        "mappingperiod": mappingperiod,
        "geomed_product_name": geomed_product_name,
        "wofs_summary_product_name": wofs_summary_product_name,
        "ard_product_names": ard_product_names,
        "region_id": region_id,
        "output": output,
        "task_table": task_table,
        "DEA Burn Cube": version,
        "summary_datasets": summary_datasets,
        "ard_datasets": ard_datasets,
    }


def odc_uuid(
    algorithm: str,
    algorithm_version: str,
    sources: Sequence[UUID],
    deployment_id: str = "",
    **other_tags,
) -> UUID:
    """
    Generate deterministic UUID for a derived Dataset.

    :param algorithm: Name of the algorithm
    :param algorithm_version: Version string of the algorithm
    :param sources: Sequence of input Dataset UUIDs
    :param deployment_id: Some sort of identifier for installation that performs
                          the run, for example Docker image hash, or dea module version on NCI.
    :param **other_tags: Any other identifiers necessary to uniquely identify dataset
    """

    tags = [f"{k}={str(v)}" for k, v in other_tags.items()]

    stringified_sources = (
        [str(algorithm), str(algorithm_version), str(deployment_id)]
        + sorted(tags)
        + [str(u) for u in sorted(sources)]
    )

    srcs_hashes = "\n".join(s.lower() for s in stringified_sources)
    return uuid5(ODC_NS, srcs_hashes)


@dataclass
class BurnCubeInputProducts:
    platform: str
    geomed: str
    wofs_summary: str
    ard_product_names: Tuple[str, ...]
    input_ard_bands: Tuple[str, ...]
    input_gm_bands: Tuple[str, ...]


@dataclass
class BurnCubeProduct:
    name: str
    short_name: str
    version: str
    product_family: str
    bands: Tuple[str, ...]


@dataclass
class BurnCubeProcessingTask:
    output_folder: str
    input_products: BurnCubeInputProducts
    product: BurnCubeProduct
    task_id: str
    region_id: str
    task_table: str
    local_file_path: str = field(init=False, repr=False)
    s3_key_path: str = field(init=False, repr=False)
    bucket_name: str = field(init=False, repr=False)
    period_start: str = field(init=False, repr=False)
    period_end: str = field(init=False, repr=False)
    mapping_period_start: str = field(init=False, repr=False)
    mapping_period_end: str = field(init=False, repr=False)

    def __post_init__(self):
        # Generate output filenames for the task
        (
            self.local_file_path,
            self.s3_key_path,
            self.bucket_name,
        ) = generate_output_filenames(
            self.output_folder,
            self.task_id,
            self.region_id,
            self.input_products.platform,
        )

        # Generate task processing periods
        processing_period = generate_task(self.task_id, self.task_table)

        self.period_start = processing_period["Period Start"]
        self.period_end = processing_period["Period End"]
        self.mapping_period_start = processing_period["Mapping Period Start"]
        self.mapping_period_end = processing_period["Mapping Period End"]

    @classmethod
    def from_config(cls, cfg_url: str, task_id: str, region_id: str):
        # Load configuration from a remote YAML file
        cfg = load_yaml_remote(cfg_url)

        input_products = BurnCubeInputProducts(**cfg["input_products"])
        product = BurnCubeProduct(**cfg["product"])

        return cls(
            output_folder=cfg["output_folder"],
            task_table=cfg["task_table"],
            input_products=input_products,
            product=product,
            task_id=task_id,
            region_id=region_id,
        )
