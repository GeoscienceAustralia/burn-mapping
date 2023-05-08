"""
This module is used to create and manage tasks for burn mapping
using the DEA Burn Cube.

"""

import calendar
import datetime
import itertools
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import timezone
from typing import Any, Dict, List, Sequence, Tuple
from urllib.parse import urlparse
from uuid import UUID, uuid5

import boto3
import botocore
import datacube
import fsspec
import pandas as pd
import pystac
import s3fs
import yaml
from datacube.utils.dates import normalise_dt
from odc.dscache.tools.tiling import parse_gridspec_with_name
from pystac.extensions.eo import Band, EOExtension
from pystac.extensions.projection import ProjectionExtension

import dea_burn_cube.__version__ as version
from dea_burn_cube import io

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# Some random UUID to be ODC namespace
# copy paste from: odc-stats Model.py design
ODC_NS = UUID("6f34c6f4-13d6-43c0-8e4e-42b6c13203af")


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


class IncorrectInputDataError(Exception):
    """
    Exception raised when the input data provided to a function or method is incorrect.

    Attributes:
    - message (str): the error message associated with the exception.

    Methods:
    - log_error(): logs the error message to a logger object.

    Usage example:
    >>> try:
    ...     # some code that might raise an IncorrectInputDataError
    ... except IncorrectInputDataError as e:
    ...     e.log_error()
    """

    def __init__(self, message):
        super().__init__(message)

    def log_error(self):
        """
        Logs the error message to a logger object.
        """
        logger.error(str(self))


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


def _get_gpgon(
    region_id: str,
) -> Tuple[datacube.utils.geometry.Geometry, datacube.utils.geometry._base.GeoBox]:
    """
    Get a geometry that covers the specified region for use with datacube.load().

    Parameters
    ----------
    region_id : str
        The ID of the region to get a geometry for. E.g. x30y29

    Returns
    -------
    Tuple[datacube.utils.geometry.Geometry, datacube.utils.geometry._base.GeoBox]
        The geometry object representing the region specified by `region_id` and the corresponding geobox.
    """

    _, gridspec = parse_gridspec_with_name("au-30")

    # gridspec : au-30
    pattern = r"x(\d+)y(\d+)"

    match = re.match(pattern, region_id)

    if match:
        x = int(match.group(1))
        y = int(match.group(2))
    else:
        logger.error(
            "No match found in region id %s.",
            region_id,
        )
        # cannot extract geobox, so we stop here.
        # if we throw exception, it will trigger the Airflow/Argo retry.
        sys.exit(0)

    geobox = gridspec.tile_geobox((x, y))

    # Return the resulting Geometry object
    return datacube.utils.geometry.Geometry(geobox.extent.geom, crs="epsg:3577"), geobox


@dataclass
class BurnCubeInputProducts:
    platform: str
    geomed: str
    wofs_summary: str
    ard_product_names: List[str]
    input_ard_bands: List[str]
    input_gm_bands: List[str]

    def validate(self):
        if not isinstance(self.platform, str):
            raise ValueError("platform must be a string")
        if not isinstance(self.geomed, str):
            raise ValueError("geomed must be a string")
        if not isinstance(self.wofs_summary, str):
            raise ValueError("wofs_summary must be a string")
        if not isinstance(self.ard_product_names, list) or not all(
            isinstance(name, str) for name in self.ard_product_names
        ):
            raise ValueError("ard_product_names must be a tuple of strings")
        if not isinstance(self.input_ard_bands, list) or not all(
            isinstance(band, str) for band in self.input_ard_bands
        ):
            raise ValueError("input_ard_bands must be a tuple of strings")
        if not isinstance(self.input_gm_bands, list) or not all(
            isinstance(band, str) for band in self.input_gm_bands
        ):
            raise ValueError("input_gm_bands must be a tuple of strings")


@dataclass
class BurnCubeProduct:
    name: str
    short_name: str
    version: str
    product_family: str
    bands: List[str]

    def validate(self):
        if not isinstance(self.name, str):
            raise ValueError("name must be a string")
        if not isinstance(self.short_name, str):
            raise ValueError("short_name must be a string")
        if not isinstance(self.version, str):
            raise ValueError("version must be a string")
        if not isinstance(self.product_family, str):
            raise ValueError("product_family must be a string")
        if not isinstance(self.bands, list) or not all(
            isinstance(band, str) for band in self.bands
        ):
            raise ValueError("bands must be a tuple of strings")


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
    s3_file_path: str = field(init=False, repr=False)
    bucket_name: str = field(init=False, repr=False)
    period_start: str = field(init=False, repr=False)
    period_end: str = field(init=False, repr=False)
    mapping_period_start: str = field(init=False, repr=False)
    mapping_period_end: str = field(init=False, repr=False)
    gpgon: datacube.utils.geometry.Geometry = field(init=False, repr=False)
    geobox: datacube.utils.geometry._base.GeoBox = field(init=False, repr=False)

    # get the following information by HNRD-DC and ODC
    geomed_datasets: List[datacube.model.Dataset] = field(init=False, repr=False)
    wofs_datasets: List[datacube.model.Dataset] = field(init=False, repr=False)
    ref_ard_datasets: List[datacube.model.Dataset] = field(init=False, repr=False)
    mapping_ard_datasets: List[datacube.model.Dataset] = field(init=False, repr=False)

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

        self.s3_file_path = f"{self.bucket_name}/{self.s3_key_path}"

        # Generate task processing periods
        processing_period = generate_task(self.task_id, self.task_table)

        self.period_start = processing_period["Period Start"]
        self.period_end = processing_period["Period End"]
        self.mapping_period_start = processing_period["Mapping Period Start"]
        self.mapping_period_end = processing_period["Mapping Period End"]

        self.gpgon, self.geobox = _get_gpgon(self.region_id)

    def validate_cfg(self):
        if not isinstance(self.output_folder, str):
            raise ValueError("output_folder must be a string")
        if not isinstance(self.task_table, str):
            raise ValueError("task_table must be a string")
        if not isinstance(self.task_id, str):
            raise ValueError("task_id must be a string")
        if not isinstance(self.region_id, str):
            raise ValueError("region_id must be a string")

        if not (
            len(self.region_id) == 6
            and self.region_id[0] == "x"
            and self.region_id[3] == "y"
            and self.region_id[1:3].isdigit()
            and self.region_id[4:6].isdigit()
        ):
            raise ValueError("region_id must be in the format 'x12y23' or 'x02y10'")

        task_id_pattern = r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{2}$"
        if not re.match(task_id_pattern, self.task_id):
            raise ValueError(
                "task_id must be in the format 'Dec-19', 'Apr-20', 'Mar-19'"
            )

        self.input_products.validate()
        self.product.validate()

    def validate_data(self):
        # The following variables passed by K8s Pod manifest
        odc_dc = datacube.Datacube(
            app=f"Burn Cube K8s processing - {self.region_id}",
            config={
                "db_hostname": os.getenv("ODC_DB_HOSTNAME"),
                "db_password": os.getenv("ODC_DB_PASSWORD"),
                "db_username": os.getenv("ODC_DB_USERNAME"),
                "db_port": 5432,
                "db_database": os.getenv("ODC_DB_DATABASE"),
            },
        )
        hnrs_dc = datacube.Datacube(
            app=f"Burn Cube K8s processing - {self.region_id}",
            config={
                "db_hostname": os.getenv("HNRS_DB_HOSTNAME"),
                "db_password": os.getenv("HNRS_DC_DB_PASSWORD"),
                "db_username": os.getenv("HNRS_DC_DB_USERNAME"),
                "db_port": 5432,
                "db_database": os.getenv("HNRS_DC_DB_DATABASE"),
            },
        )

        self.geomed_datasets = hnrs_dc.find_datasets(
            product=self.input_products.geomed,
            geopolygon=self.gpgon,
            time=self.period_start,
        )

        if len(self.geomed_datasets) != 1:
            raise IncorrectInputDataError(
                "Found " + len(self.geomed_datasets) + " GeoMAD dataset"
            )

        self.wofs_datasets = odc_dc.find_datasets(
            product=self.input_products.wofs_summary,
            geopolygon=self.gpgon,
            time=self.mapping_period_start,
        )

        if len(self.wofs_datasets) != 1:
            raise IncorrectInputDataError(
                "Found " + len(self.wofs_datasets) + " WOfS dataset"
            )

        self.ref_ard_datasets = odc_dc.find_datasets(
            product=self.input_products.ard_product_names,
            geopolygon=self.gpgon,
            time=(self.period_start, self.period_end),
        )

        if len(self.ref_ard_datasets) < 1:
            raise IncorrectInputDataError(
                "Found Any ARD dataset in " + str(self.period_start, self.period_end)
            )

        self.mapping_ard_datasets = odc_dc.find_datasets(
            product=self.input_products.ard_product_names,
            geopolygon=self.gpgon,
            time=(self.mapping_period_start, self.mapping_period_end),
        )

        if len(self.mapping_ard_datasets) < 1:
            raise IncorrectInputDataError(
                "Found Any ARD dataset in "
                + str(self.mapping_period_start, self.mapping_period_end)
            )

    def upload_processing_log(self):
        processing_log = {
            "task_id": self.task_id,
            "period": (self.period_start, self.period_end),
            "mappingperiod": (self.mapping_period_start, self.mapping_period_end),
            "geomed_product_name": self.input_products.geomed,
            "wofs_summary_product_name": self.input_products.wofs_summary,
            "ard_product_names": self.input_products.ard_product_names,
            "region_id": self.region_id,
            "output": self.s3_file_path,
            "task_table": self.task_table,
            "DEA Burn Cube": version,
            "summary_datasets": [e.metadata_doc["label"] for e in self.geomed_datasets]
            + [e.metadata_doc["label"] for e in self.wofs_datasets],
            "ard_datasets": [str(e.id) for e in self.ref_ard_datasets]
            + [str(e.id) for e in self.mapping_ard_datasets],
        }

        io.upload_dict_to_s3(
            processing_log,
            self.bucket_name,
            self.s3_key_path.replace(".nc", ".json"),
        )

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

    def add_metadata(self):

        geobox_wgs84 = self.geobox.extent.to_crs(
            "epsg:4326", resolution=math.inf, wrapdateline=True
        )

        bbox = geobox_wgs84.boundingbox

        # Note: we only pass mapping ard as Burn Cube upstream data for now
        input_datasets = self.mapping_ard_datasets

        properties: Dict[str, Any] = {}

        properties[
            "title"
        ] = f"BurnMapping-{self.input_products.platform}-{self.task_id}-{self.region_id}"
        properties["dtr:start_datetime"] = format_datetime(self.mapping_period_start)
        properties["dtr:end_datetime"] = format_datetime(self.mapping_period_end)
        properties["odc:processing_datetime"] = format_datetime(
            datetime.datetime.utcnow(), timespec="seconds"
        )
        properties["odc:region_code"] = self.region_id
        properties["odc:product"] = self.product.name
        properties["instrument"] = list(
            itertools.chain(
                *[
                    v.lower().split("_")
                    for v in {e.metadata.instrument for e in input_datasets}
                ]
            )
        )
        properties["gsd"] = [e.metadata.eo_gsd for e in input_datasets][0]
        properties["platform"] = "_".join(sorted(
            {e.metadata.platform for e in input_datasets}
        ))
        properties["odc:file_format"] = "GeoTIFF"
        properties["odc:product_family"] = self.product.product_family
        properties["odc:producer"] = "ga.gov.au"
        properties["odc:dataset_version"] = self.product.version
        properties["dea:dataset_maturity"] = "final"
        properties["odc:collection_number"] = 3

        uuid = odc_uuid(
            self.product.name,
            self.product.version,
            sources=[str(e.id) for e in input_datasets],
            tile=self.region_id,
            time=str((self.mapping_period_start, self.mapping_period_end)),
        )

        item = pystac.Item(
            id=str(uuid),
            geometry=geobox_wgs84.json,
            bbox=[bbox.left, bbox.bottom, bbox.right, bbox.top],
            datetime=pd.Timestamp(self.mapping_period_start).replace(
                tzinfo=timezone.utc
            ),
            properties=properties,
            collection=self.product.name,
        )

        ProjectionExtension.add_to(item)
        proj_ext = ProjectionExtension.ext(item)
        proj_ext.apply(
            self.geobox.crs.epsg,
            transform=self.geobox.transform,
            shape=self.geobox.shape,
        )

        # Lineage last
        item.properties["odc:lineage"] = dict(
            inputs=[str(e.id) for e in input_datasets]
        )

        # Add all the assets
        for band_name in self.product.bands:
            asset = pystac.Asset(
                href=f"BurnMapping-{self.input_products.platform}-{self.task_id}-{self.region_id}-{band_name}.tif",
                media_type="image/tiff; application=geotiff",
                roles=["data"],
                title=band_name,
            )

            eo = EOExtension.ext(asset)
            band = Band.create(band_name)
            eo.apply(bands=[band])

            proj = ProjectionExtension.ext(asset)

            proj.apply(
                self.geobox.crs.epsg,
                transform=self.geobox.transform,
                shape=self.geobox.shape,
            )

            item.add_asset(band_name, asset=asset)

        stac_metadata_path = (
            "s3://"
            + self.bucket_name
            + "/"
            + self.s3_key_path.replace(".nc", ".stac-item.json")
        )

        print("stac_metadata_path", stac_metadata_path)

        # Add links
        item.links.append(
            pystac.Link(
                rel="product_overview",
                media_type="application/json",
                target=f"https://explorer.dea.ga.gov.au/product/{self.product.name}",
            )
        )

        item.links.append(
            pystac.Link(
                rel="collection",
                media_type="application/json",
                target=f"https://explorer.dea.ga.gov.au/stac/collections/{self.product.name}",
            )
        )

        item.links.append(
            pystac.Link(
                rel="alternative",
                media_type="text/html",
                target=f"https://explorer.dea.ga.gov.au/dataset/{str(uuid)}",
            )
        )

        item.links.append(
            pystac.Link(
                rel="self",
                media_type="application/json",
                target=stac_metadata_path,
            )
        )

        stac_metadata = item.to_dict()

        logger.info(
            "Upload STAC metadata file %s in s3.",
            stac_metadata_path,
        )

        io.upload_dict_to_s3(
            stac_metadata,
            self.bucket_name,
            self.s3_key_path.replace(".nc", ".stac-item.json"),
        )
