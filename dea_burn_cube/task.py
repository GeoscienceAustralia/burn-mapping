"""
This module is used to create and manage tasks for burn mapping
using the DEA Burn Cube.

"""

import calendar
import datetime
import io
import json
import logging
import os
import re
import shutil
import sys
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from uuid import UUID, uuid5

import datacube
import eodatasets3.stac as eo3stac
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import requests
import s3fs
from eodatasets3.assemble import DatasetAssembler, serialise
from eodatasets3.images import GridSpec
from eodatasets3.scripts.tostac import json_fallback
from odc.dscache.tools.tiling import parse_gridspec_with_name
from rasterio.crs import CRS
from shapely.geometry import Point
from shapely.ops import unary_union

import dea_burn_cube.__version__ as version
from dea_burn_cube import bc_io, helper

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# Some random UUID to be ODC namespace
# copy paste from: odc-stats Model.py design
ODC_NS = UUID("6f34c6f4-13d6-43c0-8e4e-42b6c13203af")


def generate_output_filenames(
    output: str, task_id: str, region_id: str, platform: str
) -> Tuple[str, str, str]:
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
        ('BurnMapping-123-ABC', 's3://my-bucket/my-folder/123/ABC/BurnMapping-123-ABC')
    """
    bc_output_file_path = (
        f"{task_id}/{region_id}/BurnMapping-{platform}-{task_id}-{region_id}"
    )

    title = bc_output_file_path.split("/")[-1]
    s3_bucket_name, s3_object_key = helper.extract_s3_details(
        f"{output}/{bc_output_file_path}"
    )

    return title, s3_object_key, s3_bucket_name


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
    geomed_name: str
    wofs_summary_name: str
    ard_names: List[str]
    input_ard_bands: List[str]
    input_gm_bands: List[str]

    def validate(self):
        if not isinstance(self.platform, str):
            raise ValueError("platform must be a string")
        if not isinstance(self.geomed_name, str):
            raise ValueError("geomed must be a string")
        if not isinstance(self.wofs_summary_name, str):
            raise ValueError("wofs_summary must be a string")
        if not isinstance(self.ard_names, list) or not all(
            isinstance(name, str) for name in self.ard_names
        ):
            raise ValueError("ard_names must be a tuple of strings")
        if not isinstance(self.input_ard_bands, list) or not all(
            isinstance(band, str) for band in self.input_ard_bands
        ):
            raise ValueError("input_ard_bands must be a tuple of strings")
        if not isinstance(self.input_gm_bands, list) or not all(
            isinstance(band, str) for band in self.input_gm_bands
        ):
            raise ValueError("input_gm_bands must be a tuple of strings")


@dataclass
class BurnCubeOutputProduct:
    name: str
    short_name: str
    version: str
    product_family: str
    bands: List[str]
    inherit_skip_properties: List[str]
    OUTPUT_EXT: str = "GeoTIFF"
    PRODUCER: str = "ga.gov.au"
    MATURITY: str = "final"
    COLLECTION_NUM: int = 3

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
    """
    Data class representing a Burn Cube filter task.
    """

    # the Burn Cube output file title
    title: str = field(init=False, repr=False)
    output_folder: str
    # always under self.output_folder
    ancillary_folder: str = field(init=False, repr=False)

    input_products: BurnCubeInputProducts
    output_product: BurnCubeOutputProduct

    task_id: str
    region_id: str
    task_table: str
    stac_metadata_path: str = field(init=False, repr=False)

    # output S3 bucket name
    s3_bucket_name: str = field(init=False, repr=False)
    # Burn Cube output file object key path (no extension part)
    s3_object_key: str = field(init=False, repr=False)

    # Burn Cube output file object key path (with s3:// as S3 uri format)
    s3_file_uri: str = field(init=False, repr=False)

    period_start: str = field(init=False, repr=False)
    period_end: str = field(init=False, repr=False)
    mapping_period_start: str = field(init=False, repr=False)
    mapping_period_end: str = field(init=False, repr=False)

    # geometry information comes from au-30 grid and region id
    gpgon: datacube.utils.geometry.Geometry = field(init=False, repr=False)
    geobox: datacube.utils.geometry._base.GeoBox = field(init=False, repr=False)

    # get the following information by HNRD-DC and ODC
    geomed_datasets: List[datacube.model.Dataset] = field(init=False, repr=False)
    wofs_datasets: List[datacube.model.Dataset] = field(init=False, repr=False)
    ref_ard_datasets: List[datacube.model.Dataset] = field(init=False, repr=False)
    mapping_ard_datasets: List[datacube.model.Dataset] = field(init=False, repr=False)

    PROD_INFO_EXT: str = ".proc-info.json"
    ODC_META_EXT: str = ".odc-metadata.yml"
    STAC_META_EXT: str = ".stac-item.json"
    BAND_EXT: str = ".tif"

    def __post_init__(self):
        """
        Perform post-initialization tasks.

        This method generates output filenames, sets ancillary folder,
        and retrieves task processing periods based on task ID and task table.
        It also retrieves geometric information for the given region ID.
        """
        # Generate output filenames for the task
        (
            self.title,
            self.s3_object_key,
            self.s3_bucket_name,
        ) = generate_output_filenames(
            self.output_folder,
            self.task_id,
            self.region_id,
            self.input_products.platform,
        )

        self.ancillary_folder = f"{self.output_folder}/ancillary_file"

        self.s3_file_uri = f"s3://{self.s3_bucket_name}/{self.s3_object_key}"

        self.proc_info_path = self.s3_file_uri + self.PROD_INFO_EXT
        self.odc_metadata_path = self.s3_file_uri + self.ODC_META_EXT
        self.stac_metadata_path = self.s3_file_uri + self.STAC_META_EXT

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
        self.output_product.validate()

    def validate_data(self):
        # The following variables passed by K8s Pod manifest
        odc_dc = datacube.Datacube(
            app=f"Burn Cube K8s load metadata - {self.region_id}",
            config={
                "db_hostname": os.getenv("ODC_DB_HOSTNAME"),
                "db_password": os.getenv("ODC_DB_PASSWORD"),
                "db_username": os.getenv("ODC_DB_USERNAME"),
                "db_port": 5432,
                "db_database": os.getenv("ODC_DB_DATABASE"),
            },
        )
        hnrs_dc = datacube.Datacube(
            app=f"Burn Cube K8s load metadata - {self.region_id}",
            config={
                "db_hostname": os.getenv("HNRS_DB_HOSTNAME"),
                "db_password": os.getenv("HNRS_DC_DB_PASSWORD"),
                "db_username": os.getenv("HNRS_DC_DB_USERNAME"),
                "db_port": 5432,
                "db_database": os.getenv("HNRS_DC_DB_DATABASE"),
            },
        )

        self.geomed_datasets = hnrs_dc.find_datasets(
            product=self.input_products.geomed_name,
            geopolygon=self.gpgon,
            time=self.period_start,
        )

        if len(self.geomed_datasets) != 1:
            raise IncorrectInputDataError(
                "Found " + len(self.geomed_datasets) + " GeoMAD dataset"
            )

        self.wofs_datasets = odc_dc.find_datasets(
            product=self.input_products.wofs_summary_name,
            geopolygon=self.gpgon,
            time=self.mapping_period_start,
        )

        if len(self.wofs_datasets) != 1:
            raise IncorrectInputDataError(
                "Found " + len(self.wofs_datasets) + " WOfS dataset"
            )

        self.ref_ard_datasets = odc_dc.find_datasets(
            product=self.input_products.ard_names,
            geopolygon=self.gpgon,
            time=(self.period_start, self.period_end),
        )

        if len(self.ref_ard_datasets) < 1:
            raise IncorrectInputDataError(
                "Found Any ARD dataset in " + str(self.period_start, self.period_end)
            )

        self.mapping_ard_datasets = odc_dc.find_datasets(
            product=self.input_products.ard_names,
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
            "geomed_name": self.input_products.geomed_name,
            "wofs_summary_name": self.input_products.wofs_summary_name,
            "ard_names": self.input_products.ard_names,
            "region_id": self.region_id,
            "output": self.s3_file_uri,
            "task_table": self.task_table,
            "DEA Burn Cube": version,
            "summary_datasets": [e.metadata_doc["label"] for e in self.geomed_datasets]
            + [e.metadata_doc["label"] for e in self.wofs_datasets],
            "ard_datasets": [str(e.id) for e in self.ref_ard_datasets]
            + [str(e.id) for e in self.mapping_ard_datasets],
        }

        logger.info("Upload processing log file %s in s3.", self.proc_info_path)

        local_proc_info_path = self.title + self.PROD_INFO_EXT

        with open(local_proc_info_path, "w") as proc_info_file:
            json.dump(processing_log, proc_info_file, indent=2)

        bc_io.upload_dict_to_s3(local_proc_info_path, self.proc_info_path)

    @classmethod
    def from_config(cls, cfg_url: str, task_id: str, region_id: str):
        # Load configuration from a remote YAML file
        cfg = helper.load_yaml_remote(cfg_url)

        input_products = BurnCubeInputProducts(**cfg["input_products"])
        output_product = BurnCubeOutputProduct(**cfg["product"])

        return cls(
            output_folder=cfg["output_folder"],
            task_table=cfg["task_table"],
            input_products=input_products,
            output_product=output_product,
            task_id=task_id,
            region_id=region_id,
        )

    def add_metadata_files(self):

        dataset_assembler = DatasetAssembler(
            naming_conventions="dea_c3",
            dataset_location=Path(
                f"https://explorer.dea.ga.gov.au/product/{self.output_product.name}"
            ),
            allow_absolute_paths=True,
        )

        # ignore the tons of Inheritable property warnings
        warnings.simplefilter(action="ignore", category=UserWarning)

        platforms, instruments = ([], [])

        for dataset in self.mapping_ard_datasets:

            source_datasetdoc = serialise.from_doc(
                dataset.metadata_doc, skip_validation=True
            )
            dataset_assembler.add_source_dataset(
                source_datasetdoc,
                classifier="ard",
                auto_inherit_properties=True,  # it will grab all useful input dataset preperties
                inherit_geometry=False,
                inherit_skip_properties=self.output_product.inherit_skip_properties,
            )

            if "eo:platform" in source_datasetdoc.properties:
                platforms.append(source_datasetdoc.properties["eo:platform"])
            if "eo:instrument" in source_datasetdoc.properties:
                instruments.append(source_datasetdoc.properties["eo:instrument"])

        dataset_assembler.platform = ",".join(sorted(set(platforms)))
        dataset_assembler.instrument = "_".join(sorted(set(instruments)))

        dataset_assembler.geometry = self.geobox.extent.geom

        dataset_assembler.datetime = helper.format_datetime(self.mapping_period_start)
        dataset_assembler.properties["dtr:start_datetime"] = helper.format_datetime(
            self.mapping_period_start
        )
        dataset_assembler.properties["dtr:end_datetime"] = helper.format_datetime(
            self.mapping_period_end
        )

        dataset_assembler.product_name = self.output_product.name
        dataset_assembler.dataset_version = self.output_product.version
        dataset_assembler.region_code = self.region_id

        dataset_assembler.properties["odc:file_format"] = self.output_product.OUTPUT_EXT
        dataset_assembler.properties["odc:producer"] = self.output_product.PRODUCER
        dataset_assembler.properties[
            "odc:product_family"
        ] = self.output_product.product_family

        # set the warning message back
        warnings.filterwarnings("default")

        dataset_assembler.processed = datetime.datetime.utcnow()

        dataset_assembler.maturity = self.output_product.MATURITY
        dataset_assembler.collection_number = self.output_product.COLLECTION_NUM

        for band_name in self.output_product.bands:
            dataset_assembler.note_measurement(
                band_name,
                f"{self.title}-{band_name}{self.BAND_EXT}",
                expand_valid_data=False,
                grid=GridSpec(
                    shape=self.geobox.shape,
                    transform=self.geobox.transform,
                    crs=CRS.from_epsg(self.geobox.crs.to_epsg()),
                ),
                nodata=-999,
            )
        dataset_assembler.extend_user_metadata(
            "input-products", sorted({e.type.name for e in self.mapping_ard_datasets})
        )

        dataset_assembler._accessories["metadata:processor"] = (
            self.title + self.PROD_INFO_EXT
        )

        meta = dataset_assembler.to_dataset_doc()
        # already add all information to dataset_assembler,
        # now convert to odc and stac metadata format
        stac_meta = eo3stac.to_stac_item(
            dataset=meta,
            stac_item_destination_url=self.stac_metadata_path,
            dataset_location=str(Path(self.s3_file_uri).parent),
            odc_dataset_metadata_url=self.odc_metadata_path,
            explorer_base_url=f"https://explorer.dea.ga.gov.au/product/{self.output_product.name}",
        )
        stac_meta = json.dumps(
            stac_meta,
            default=json_fallback,
            indent=4,
        )  # stac_meta is Python str
        local_stac_metadata_path = self.title + self.STAC_META_EXT

        with open(local_stac_metadata_path, "w") as json_file:
            json_file.write(stac_meta)

        logger.info("Upload STAC metadata to", self.stac_metadata_path)

        bc_io.upload_object_to_s3(local_stac_metadata_path, self.stac_metadata_path)

        meta_stream = io.StringIO("")  # too short, not worth to move to another method.
        serialise.to_stream(meta_stream, meta)
        odc_meta = meta_stream.getvalue()  # odc_meta is Python str

        local_odc_metadata_path = {self.title} + self.ODC_META_EXT

        with open(local_odc_metadata_path, "w") as yml_file:
            yml_file.write(odc_meta)

        logger.info("Upload ODC metadata to", self.odc_metadata_path)

        bc_io.upload_object_to_s3(local_odc_metadata_path, self.odc_metadata_path)


@dataclass
class BurnCubeFilterTask:
    """
    Data class representing a Burn Cube filter task.
    """

    output_folder: str
    ancillary_folder: str
    task_id: str
    region_list: str
    platform: str
    task_table: str

    region_list_local_uri: str
    region_list_s3_uri: str

    hotspot_csv_local_uri: str
    hotspot_csv_s3_uri: str

    csv_filename = "hotspot_historic.csv"
    hotspot_product_url = (
        "https://ga-sentinel.s3-ap-southeast-2.amazonaws.com/historic/all-data-csv.zip"
    )
    local_hotspot_file_path = "all-data-csv.zip"

    ocean_mask_path = (
        "s3://dea-public-data-dev/projects/burn_cube/configs/ITEMCoastlineCleaned.shp"
    )

    def __init__(self, process_cfg_url: str, task_id: str):
        """
        Initialize BurnCubeFilterTask instance.

        Args:
            process_cfg_url: The URL of the process configuration file.
            task_id: The ID of the task.

        Raises:
            IOError: If there is an error loading the process configuration file.
        """
        # Load pr
        process_cfg = helper.load_yaml_remote(process_cfg_url)

        self.task_id = task_id
        self.task_table = process_cfg["task_table"]
        self.output_folder = process_cfg["output_folder"]
        self.platform = process_cfg["input_products"]["platform"]
        self.ancillary_folder = f"{self.output_folder}/ancillary_file"

        self.region_list_local_uri = f"{self.task_id}-regions.json"
        self.region_list_s3_uri = (
            f"{self.ancillary_folder}/{self.region_list_local_uri}"
        )

        self.hotspot_csv_local_uri = f"{self.task_id}-{self.csv_filename}"
        self.hotspot_csv_s3_uri = (
            f"{self.ancillary_folder}/{self.hotspot_csv_local_uri}"
        )

    def filter_by_hotspot(self) -> pd.DataFrame:
        """
        Filter hotspot data based on the mapping period and sensor information.

        Returns:
            A Pandas DataFrame containing the filtered hotspot data.

        Raises:
            IOError: If there is an error downloading or extracting the hotspot file.
        """
        bc_running_task = generate_task(self.task_id, self.task_table)

        mappingperiod = (
            bc_running_task["Mapping Period Start"],
            bc_running_task["Mapping Period End"],
        )

        logger.info("Use mappingperiod: %s to filter hotspot file", str(mappingperiod))

        start = (
            np.datetime64(mappingperiod[0]).astype("datetime64[ns]")
            - np.datetime64(2, "M")
        ).astype("datetime64[ns]")
        stop = np.datetime64(mappingperiod[1])

        # the current (10/01/2023) zip file size is 430MB. It is safe to download it to local file system
        r = requests.get(self.hotspot_product_url, stream=True)
        r.raw.decode_content = True
        with open(self.local_hotspot_file_path, "wb") as f:
            shutil.copyfileobj(r.raw, f)

        # load the CSV file from zip file
        with zipfile.ZipFile(self.local_hotspot_file_path) as z:
            with z.open(self.csv_filename) as f:

                # only load these 4 columns from hotspot to save RAM
                column_names = ["datetime", "sensor", "latitude", "longitude"]

                # read the hotspot data as Pandas.DataFrame
                hotspot_df = pd.read_csv(f, usecols=column_names, low_memory=False)

                dates = pd.to_datetime(
                    hotspot_df.datetime.apply(lambda x: x.split("+")[0]).values
                )

                # filter the dataframe: just filter by period and sensor information
                index = np.where(
                    (hotspot_df.sensor == "MODIS") & (dates >= start) & (dates <= stop)
                )[0]

        return hotspot_df[hotspot_df.index.isin(index)]

    def filter_by_region(self, region_list_s3_path: str) -> gpd.GeoDataFrame:
        """
        Filter regions by ocean mask and hot spot.

        Args:
            region_list_s3_path: The S3 path of the region list file.

        Returns:
            A GeoDataFrame containing the filtered regions.

        Raises:
            IOError: If there is an error reading the region list file or hot spot CSV file.
        """
        _ = s3fs.S3FileSystem(anon=True)
        _ = "s3" in gpd.io.file._VALID_URLS
        gpd.io.file._VALID_URLS.discard("s3")

        region_gdf = gpd.read_file(region_list_s3_path)
        region_gdf = region_gdf.to_crs(epsg="3577")

        logger.info("Filter %s by Ocean Mask", region_list_s3_path)

        ocean_mask = gpd.read_file(self.ocean_mask_path)

        # the Ocean Mask CRS should be: EPSG:3577
        filter_by_ocean_mask = []

        for region_index in region_gdf.index:
            region_id = region_gdf.region_code[region_index]
            region_geometry = region_gdf.geometry[region_index]

            for ocean_index in ocean_mask.index:
                if region_geometry.intersects(ocean_mask.geometry[ocean_index]):
                    filter_by_ocean_mask.append(region_id)
                    break

        region_gdf = region_gdf[
            region_gdf["region_code"].isin(filter_by_ocean_mask)
        ].reindex()

        logger.info(
            "The number of region changes to %s after Ocean Mask filter",
            str(len(region_gdf)),
        )

        # we assume the formats are always same, with columns: region_code, i_x, i_y, utc_offset, geometry
        # also the geometry are always Polygon
        logger.info("Filter regions by Hot Spot %s", self.hotspot_csv_s3_uri)

        hotspot_df = pd.read_csv(self.hotspot_csv_s3_uri)
        latitude = hotspot_df.latitude.values
        longitude = hotspot_df.longitude.values

        reverse_transformer = pyproj.Transformer.from_crs("EPSG:4283", "EPSG:3577")
        easting, northing = reverse_transformer.transform(latitude, longitude)

        patch = [
            Point(easting[i], northing[i]).buffer(4000)
            for i in range(0, len(hotspot_df))
        ]
        hotspot_polygons = unary_union(patch)

        filter_by_hotspot = []

        for region_index in region_gdf.index:
            region_id = region_gdf.region_code[region_index]
            region_geometry = region_gdf.geometry[region_index]
            if region_geometry.intersects(hotspot_polygons):
                filter_by_hotspot.append(region_id)

        region_gdf = region_gdf[
            region_gdf["region_code"].isin(filter_by_hotspot)
        ].reindex()

        # shuffle the region list to aviod data skew
        region_gdf = region_gdf.sample(frac=1).reset_index(drop=True)

        logger.info(
            "The number of region changes to %s after Hot Spot filter",
            str(len(region_gdf)),
        )

        return region_gdf

    def filter_by_output(self, overwrite) -> gpd.GeoDataFrame:
        """
        Filter regions by output NetCDF files.

        Returns:
            A GeoDataFrame containing the regions that do not have corresponding NetCDF files in S3.
        """

        _ = s3fs.S3FileSystem(anon=True)

        _ = "s3" in gpd.io.file._VALID_URLS
        gpd.io.file._VALID_URLS.discard("s3")

        region_gdf = gpd.read_file(self.region_list_s3_uri)
        region_gdf = region_gdf.to_crs(epsg="3577")

        if overwrite:
            logger.info(
                "Overwrite mode, no need filter %s by output NetCDF files",
                self.region_list_s3_uri,
            )
            not_run_geojson = region_gdf
        else:
            logger.info("Filter %s by output NetCDF files", self.region_list_s3_uri)

            not_run_regions: List[str] = []
            for region_index in region_gdf.index:
                region_id = region_gdf.region_code[region_index]

                _, s3_object_title, s3_bucket_name = generate_output_filenames(
                    self.output_folder, self.task_id, region_id, self.platform
                )

                if not helper.check_s3_file_exists(
                    f"s3://{s3_bucket_name}/{s3_object_title}.odc-metadata.yml"
                ):
                    not_run_regions.append(region_id)

            not_run_geojson = region_gdf[
                region_gdf["region_code"].isin(not_run_regions)
            ].reindex()

        return not_run_geojson
