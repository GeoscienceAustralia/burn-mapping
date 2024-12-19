# Import modules
import logging
import os
import sys

import click
import datacube
import joblib
import requests
import xarray as xr
from datacube.utils.cog import write_cog
from dea_tools.classification import sklearn_flatten, sklearn_unflatten
from odc.algo import mask_cleanup

from dea_burn_cube import bc_io, helper

# Configure logging
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def logging_setup():
    """Set up logging for all modules except sqlalchemy and boto."""
    loggers = [
        logging.getLogger(name)
        for name in logging.root.manager.loggerDict
        if not name.startswith("sqlalchemy") and not name.startswith("boto")
    ]

    stdout_hdlr = logging.StreamHandler(sys.stdout)
    for logger in loggers:
        logger.addHandler(stdout_hdlr)
        logger.propagate = False


def classify_fmc(data, model):
    """
    Perform FMC classification using a pre-trained model.

    Args:
        data (xarray.Dataset): Sentinel-2 dataset with required bands and optional multiple time steps.
        model (sklearn.BaseEstimator): A pre-trained scikit-learn model for classification.

    Returns:
        xarray.Dataset: Dataset containing the classified FMC results.
    """
    # Calculate NDVI and NDII
    data["ndii"] = (data.nbart_nir_1 - data.nbart_swir_2) / (
        data.nbart_nir_1 + data.nbart_swir_2
    )
    data["ndvi"] = (data.nbart_nir_1 - data.nbart_red) / (
        data.nbart_nir_1 + data.nbart_red
    )

    # Reorder variables to match model expectations
    bands_order = [
        "ndvi",
        "ndii",
        "nbart_blue",
        "nbart_green",
        "nbart_red",
        "nbart_red_edge_1",
        "nbart_red_edge_2",
        "nbart_red_edge_3",
        "nbart_nir_1",
        "nbart_nir_2",
        "nbart_swir_2",
        "nbart_swir_3",
    ]
    data_neworder = data[bands_order]

    # Flatten the data for classification
    data_flat = sklearn_flatten(data_neworder)

    # Classify the data
    out_class = model.predict(data_flat)

    # Unflatten the classified data to original shape
    returned_result = sklearn_unflatten(out_class, data).transpose()

    # Create the output dataset
    dataset_result = xr.Dataset(
        {"LFMC": returned_result}, coords=data.coords, attrs=data.attrs
    )

    return dataset_result


def download_file_from_s3_public(url, file_path):
    """Download a file from a public S3 URL."""
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
        logger.info(f"File downloaded successfully from: {url}")
    else:
        logger.error(f"Failed to download file from: {url}")


@click.command(no_args_is_help=True)
@click.option(
    "--dataset-uuid",
    "-d",
    type=str,
    required=True,
    help="Region ID (AU-30 Grid).",
)
@click.option(
    "--process-cfg-url",
    "-p",
    type=str,
    required=True,
    help="URL to Burn Cube process configuration file in YAML format.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Rerun scenes that have already been processed.",
)
def fmc_processing(dataset_uuid, process_cfg_url, overwrite):
    """
    Process FMC for a given dataset UUID and configuration.

    Args:
        dataset_uuid (str): UUID of the dataset to process.
        process_cfg_url (str): URL to the process configuration YAML file.
        overwrite (bool): Whether to overwrite existing results.
    """
    logging_setup()

    # Initialize Datacube instance
    dc = datacube.Datacube(
        app="fmc_processing",
        config={
            "db_hostname": os.getenv("ODC_DB_HOSTNAME"),
            "db_password": os.getenv("ODC_DB_PASSWORD"),
            "db_username": os.getenv("ODC_DB_USERNAME"),
            "db_port": 5432,
            "db_database": os.getenv("ODC_DB_DATABASE"),
        },
    )

    # Set AWS credentials for accessing public data
    os.environ["AWS_NO_SIGN_REQUEST"] = "Yes"

    # Load process configuration
    process_cfg = helper.load_yaml_remote(process_cfg_url)
    measurements_list = process_cfg["input_products"]["input_bands"]
    output_folder = process_cfg["output_folder"]
    model_url = process_cfg["model_path"]
    product_name = process_cfg["product"]["name"]
    product_version = str(process_cfg["product"]["version"]).replace(".", "-")

    # Download the pre-trained model
    model_path = "RF_AllBands_noLC_DEA_labeless.joblib"
    download_file_from_s3_public(model_url, model_path)
    model = joblib.load(model_path)

    # Load the dataset from Datacube
    dataset = dc.index.datasets.get(dataset_uuid)
    df = dc.load(
        datasets=[dataset],
        measurements=measurements_list,
        resolution=(-20, 20),
        resampling={"*": "bilinear"},
        output_crs="EPSG:3577",
    )

    # Create masks
    cloud_mask = (df.oa_fmask == 2) | (df.oa_fmask == 3)
    water_mask = (df.oa_fmask == 5) | (df.oa_fmask == 0)
    better_cloud_mask = mask_cleanup(
        mask=cloud_mask, mask_filters=[("opening", 1), ("dilation", 3)]
    )

    # Drop unnecessary variables
    df = df.drop_vars(["oa_fmask"])

    # Perform FMC classification
    fmc_data = classify_fmc(df, model)

    # Apply masks to classified data
    masked_data = fmc_data.where(~better_cloud_mask & ~water_mask)
    masked_data = masked_data.where(masked_data >= 0, -999).astype("int16")

    # Define output file details
    region_code = dataset.metadata.fields["region_code"]
    acquisition_date = dataset.metadata.fields["time"][0].date().strftime("%Y-%m-%d")
    local_tif = f"{product_name}_{region_code}_{acquisition_date}_fmc.tif"

    # Save results to GeoTIFF
    write_cog(masked_data.LFMC, fname=local_tif, overwrite=True, nodata=-999)
    logger.info(f"Result saved as: {local_tif}")

    # Upload result to S3
    s3_folder = (
        f"{output_folder}/{product_name}/{product_version}/{region_code[:2]}/{region_code[2:]}/"
        + acquisition_date.replace("-", "/")
    )
    s3_file_uri = f"{s3_folder}/{local_tif}"

    helper.get_and_set_aws_credentials()
    bc_io.upload_object_to_s3(local_tif, s3_file_uri)
    logger.info(f"Uploaded result to: {s3_file_uri}")


if __name__ == "__main__":
    fmc_processing()
