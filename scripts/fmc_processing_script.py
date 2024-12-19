# import modules
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

logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def logging_setup():
    """Set up logging."""
    loggers = [
        logging.getLogger(name)
        for name in logging.root.manager.loggerDict
        if not name.startswith("sqlalchemy") and not name.startswith("boto")
    ]

    stdout_hdlr = logging.StreamHandler(sys.stdout)
    for logger in loggers:
        logger.addHandler(stdout_hdlr)
        logger.propagate = False


def classify_FMC(data, model):
    """
    - data (xarray.Dataset): Sentinel-2 dataset containing required bands and optional multiple time steps.

    - model (sklearn model): A pre-trained model for classification.

    Returns:
    - xarray.Dataset: A dataset containing the classified FMC results."""
    # calculate NDVI and NDII

    data["ndii"] = (data.nbart_nir_1 - data.nbart_swir_2) / (
        data.nbart_nir_1 + data.nbart_swir_2
    )
    data["ndvi"] = (data.nbart_nir_1 - data.nbart_red) / (
        data.nbart_nir_1 + data.nbart_red
    )

    # change order of variables to be the same as the model expects
    data_neworder = data[
        [
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
    ]

    # flattern the data using SKlearn_flatten
    data_flat = sklearn_flatten(data_neworder)

    # classify the data using the model
    # print("predicting...")
    out_class = model.predict(data_flat)

    # return_classification to original shape
    # transpose because coords sideways when moving from Numpy to Xarray
    returned_result = sklearn_unflatten(out_class, data).transpose()

    # make results a dataset
    dataset_result = xr.Dataset(
        {"LFMC": returned_result}, coords=data.coords, attrs=data.attrs
    )

    # return
    return dataset_result


def download_file_from_s3_public(url, file_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"File downloaded successfully from: {url}")
    else:
        print(f"Failed to download file from: {url}")


@click.command(no_args_is_help=True)
@click.option(
    "--dataset-uuid",
    "-d",
    type=str,
    default=None,
    help="REQUIRED. Region id AU-30 Grid.",
)
@click.option(
    "--process-cfg-url",
    "-p",
    type=str,
    default=None,
    help="REQUIRED. The Path URL to Burn Cube process cfg file as YAML format.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Rerun scenes that have already been processed.",
)
def fmc_processing(
    dataset_uuid,
    process_cfg_url,
    overwrite,
):
    logging_setup()

    dc = datacube.Datacube(
        app=f"fmc_processing",
        config={
            "db_hostname": os.getenv("ODC_DB_HOSTNAME"),
            "db_password": os.getenv("ODC_DB_PASSWORD"),
            "db_username": os.getenv("ODC_DB_USERNAME"),
            "db_port": 5432,
            "db_database": os.getenv("ODC_DB_DATABASE"),
        },
    )

    # need to set the AWS login so that we can access the data we need
    os.environ["AWS_NO_SIGN_REQUEST"] = "Yes"

    process_cfg = helper.load_yaml_remote(process_cfg_url)

    measurements_list = process_cfg["input_products"]["input_bands"]
    output_folder = process_cfg["output_folder"]
    model_url = process_cfg["model_path"]
    product_name = process_cfg["product"]["name"]
    product_version = str(process_cfg["product"]["version"]).replace(".", "-")

    # Define the path to the saved machine learning model file.
    model_path = "RF_AllBands_noLC_DEA_labeless.joblib"

    # auto download Machine Learning model from AWS S3

    download_file_from_s3_public(model_url, model_path)

    # import model: move it to fmc processing cfg file
    model = joblib.load(model_path)

    # find a single S2 dataset: 26cce90f-c8f9-4234-835c-35005454f62b

    df = dc.load(
        datasets=[dc.index.datasets.get(dataset_uuid)],
        measurements=measurements_list,
        resolution=(-20, 20),
        resampling={"*": "bilinear"},
        output_crs="EPSG:3577",
    )

    # Define masks. seperate cloud + shadow mask from water+ no_data mask becasue we want to do buffering of could+shadow but not water+no_data
    cloud_mask = (df.oa_fmask == 2) | (df.oa_fmask == 3)
    water_mask = (df.oa_fmask == 5) | (df.oa_fmask == 0)

    # perfrom 1 pixel opening on cloud + shadow. three pixle dilation
    better_cloud_mask = mask_cleanup(
        mask=cloud_mask, mask_filters=[("opening", 1), ("dilation", 3)]
    )

    # drop fmask from dataset before we classify
    df = df.drop(["oa_fmask"])

    # perfrom calssification of data usinf model defined above
    FMC_data = classify_FMC(df, model)

    # apply masks we generated before to classified data. it can be masked before we classify but then these pixels have a 0 value and it is better if it is 'no data'
    masked_data = FMC_data.where(~better_cloud_mask)
    masked_data = masked_data.where(~water_mask)

    # Make the no data be -999 not np.nan
    masked_data = masked_data.where(masked_data >= 0, -999).astype("int16")

    region_code = dc.index.datasets.get(dataset_uuid).metadata.fields["region_code"]
    nm_date = (
        dc.index.datasets.get(dataset_uuid)
        .metadata.fields["time"][0]
        .date()
        .strftime("%Y-%m-%d")
    )

    local_tif = product_name + f"_{region_code}_{nm_date}_fmc.tif"

    # save to file
    write_cog(masked_data.LFMC, fname=local_tif, overwrite=True, nodata=-999)

    logger.info("Save result as: " + str(local_tif))

    s3_file_uri = f"{output_folder}/{product_name}/{product_version}/{region_code[:3]}/{region_code[3:]}/{local_tif}"

    logger.info("Upload result to AWS S3 file: " + str(s3_file_uri))

    # activate AWS credential from attached service account
    helper.get_and_set_aws_credentials()

    bc_io.upload_object_to_s3(local_tif, s3_file_uri)

    logger.info("finish proection object filter: " + str(dataset_uuid))


if __name__ == "__main__":
    fmc_processing()
