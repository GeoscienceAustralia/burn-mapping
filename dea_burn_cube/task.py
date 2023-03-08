import calendar
import datetime
import logging
import time
from typing import Any, Dict, List

import pandas as pd
import s3fs

import dea_burn_cube.__version__ as version

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
        logging.info(f"{func.__name__} took {duration:.2f} seconds to execute.")
        return result

    return wrapper


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
    input_dataset_list: List[Dict[str, Any]],
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
    input_dataset_list : List[Dict[str, Any]]
        List of input datasets with metadata

    Returns:
    processing_log : Dict[str, Any]): A dictionary containing processing log information
    """
    processing_log: Dict[str, Any] = {}
    processing_log["task_id"] = task_id
    processing_log["period"] = period
    processing_log["mappingperiod"] = mappingperiod
    processing_log["geomed_product_name"] = geomed_product_name
    processing_log["wofs_summary_product_name"] = wofs_summary_product_name
    processing_log["ard_product_names"] = ard_product_names
    processing_log["region_id"] = region_id
    processing_log["output"] = region_id
    processing_log["task_table"] = task_table
    processing_log["input_dataset_list"] = input_dataset_list

    processing_log["DEA Burn Cube"] = version

    return processing_log