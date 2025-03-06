import hashlib
import logging
import os
import re
import sys
from typing import Tuple

import click
import datacube
import numpy as np
import requests
import rioxarray
import xarray as xr
from datacube.utils.cog import write_cog
from dea_tools.bandindices import calculate_indices
from dea_tools.classification import predict_xr
from joblib import load
from odc.dscache.tools.tiling import parse_gridspec_with_name
from scipy import ndimage
from scipy.ndimage._measurements import _stats
from skimage import morphology
from skimage.segmentation import quickshift

from dea_burn_cube import bc_io, helper, task

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

    x = int(match.group(1))
    y = int(match.group(2))

    geobox = gridspec.tile_geobox((x, y))

    # Return the resulting Geometry object
    return datacube.utils.geometry.Geometry(geobox.extent.geom, crs="epsg:3577"), geobox


# Define the feature_layers function
# This function generates the data required by the RF model to map burnt area
def feature_layers(
    query,
    hnrs_dc,
    dc,
    time_pre,
    time_post,
    climate_dataset,
    pre_fire_gm_product_name,
    post_geomed_name,
    region_id,
):

    geomed_datasets = dc.find_datasets(
        product=post_geomed_name,
        geopolygon=query["geopolygon"],
        time=time_post[0].split("-")[0],
    )

    if len(geomed_datasets) == 0:
        logger.info(f"Cannot find 1 Year GM dataset: {region_id}")
        sys.exit(0)

    ds_post = dc.load(post_geomed_name, time=time_post[0].split("-")[0], **query)

    # Dictionary mapping old variable names to new ones
    rename_dict = {
        "nbart_blue": "blue",
        "nbart_green": "green",
        "nbart_red": "red",
        "nbart_nir": "nir",
        "nbart_swir_1": "swir1",
        "nbart_swir_2": "swir2",
    }

    ds_post = ds_post.rename(rename_dict)

    base_measurements = ["nbart_blue", "nbart_green", "nbart_red", "nbart_nir", "nbart_swir_1", "nbart_swir_2"]
    # query['measurements'] = base_measurements
    del query["measurements"]

    geomed_datasets = hnrs_dc.find_datasets(
        product=pre_fire_gm_product_name,
        geopolygon=query["geopolygon"],
        time=time_pre,
    )

    if len(geomed_datasets) == 0:
        logger.info(f"Cannot find 4 Year GM dataset: {region_id}")
        sys.exit(0)

    # Load ls8 geomedians
    ds_base = hnrs_dc.load(product=pre_fire_gm_product_name, time=time_pre, **query)
    ds_base = ds_base[base_measurements]
    ds_base = ds_base.rename(rename_dict)

    # Load Land Cover
    # NOTE: the ga_ls_landcover_class_cyear_3 is 2025 LC version
    lc_query = query
    lc_query["measurements"] = ["level3", "level4"]

    ds_lc = dc.load("ga_ls_landcover_class_cyear_3", time=time_post, **query)

    # the landcover level 3 and level 4 should convert to one-hot encoding data.

    # level 3
    # 0: No data
    # 111: Cultivated Terrestrial Vegetation (CTV)
    # 112: (Semi-)Natural Terrestrial Vegetation (NTV)
    # 124: Natural Aquatic Vegetation (NAV)
    # 215: Artificial Surface (AS)
    # 216: Natural Bare Surface (NS)
    # 220: Water

    for level3_key in [0, 111, 112, 124, 215, 216, 220]:
        level3_key_name = f"level3_{str(level3_key)}"
        ds_lc[level3_key_name] = xr.where(ds_lc["level3"] == level3_key, 1, 0)

    # Drop the original 'level3' variable
    ds_lc = ds_lc.drop_vars("level3")

    # level 4
    # refs to detail table here: https://knowledge.dea.ga.gov.au/data/product/dea-land-cover-landsat/?tab=details

    level4_keys = [
        0,
        1,
        3,
        4,
        5,
        6,
        7,
        8,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        104,
    ]

    for level4_key in level4_keys:
        level4_key_name = f"level4_{str(level4_key)}"
        ds_lc[level4_key_name] = xr.where(ds_lc["level4"] == level4_key, 1, 0)

    # Drop the original 'level4' variable
    ds_lc = ds_lc.drop_vars("level4")

    ds_lc = ds_lc.isel(time=0)
    ds_lc = ds_lc.drop_vars("time")

    # Calculate band indices for pre and post-fire data
    # Calculate the base(pre) indices
    da_base = calculate_indices(
        ds_base, index=["NDVI", "NBR", "NDMI"], drop=False, collection="ga_gm_3"
    )
    da_base["VARI_pre"] = (ds_base.green - ds_base.red) / (
        ds_base.green + ds_base.red - ds_base.blue
    )

    # Renaming these indices for clarity with '_pre' suffixes
    da_base = da_base.rename({"NDVI": "NDVI_pre", "NDMI": "NDMI_pre", "NBR": "NBR_pre"})

    # Calculate the post indices
    da_post = calculate_indices(
        ds_post, index=["NDVI", "NBR", "NDMI"], drop=False, collection="ga_ls_3"
    )
    da_post["VARI_post"] = (ds_post.green - ds_post.red) / (
        ds_post.green + ds_post.red - ds_post.blue
    )
    da_post["BAI_post"] = 1 / (((0.1 - ds_post.red) ** 2) + ((0.06 - ds_post.nir) ** 2))
    # Renaming these indices for clarity with '_post' suffixes
    da_post = da_post.rename(
        {"NDVI": "NDVI_post", "NDMI": "NDMI_post", "NBR": "NBR_post"}
    )

    # Calculate differences in some indices between pre and post-fire data
    dndvi = da_base.NDVI_pre.isel(time=0) - da_post.NDVI_post
    dndvi = dndvi.rename("dNDVI")
    dnbr = da_base.NBR_pre.isel(time=0) - da_post.NBR_post
    dnbr = dnbr.rename("dNBR")
    dndmi = da_base.NDMI_pre.isel(time=0) - da_post.NDMI_post
    dndmi = dndmi.rename("dNDMI")
    dvari = da_base.VARI_pre.isel(time=0) - da_post.VARI_post
    dvari = dvari.rename("dVARI")

    # Remove unnecessary variables from the datasets
    drop_list = ["green", "red", "blue", "nir", "swir1", "swir2"]
    da_base = da_base.drop_vars(drop_list)
    da_base = da_base.isel(time=0)
    da_base = da_base.drop_vars("time")
    da_post = da_post.drop_vars(drop_list)

    # Extract climate data based on the specified geographical polygon (query_pgon)
    query_pgon = query["geopolygon"]
    x_range = query_pgon.boundingbox.range_x
    y_range = query_pgon.boundingbox.range_y
    ds_climate = climate_dataset.sel(
        x=slice(x_range[0], x_range[1]), y=slice(y_range[1], y_range[0])
    )

    # Reproject the climate data to match the post-fire data's CRS
    da_post = da_post.rio.write_crs("EPSG:3577")
    ds_climate = ds_climate.rio.write_crs("EPSG:3577")
    ds_climate = ds_climate.rio.reproject_match(da_post)

    # Create new climate code variables based on the value of 'climate_code'
    ds_climate["climate_code_1"] = ds_climate["climate_code"] * 0
    ds_climate["climate_code_2"] = ds_climate["climate_code"] * 0
    ds_climate["climate_code_3"] = ds_climate["climate_code"] * 0

    ds_climate["climate_code_1"] = xr.where(
        ds_climate["climate_code"] == 1, 1, ds_climate["climate_code_1"]
    )
    ds_climate["climate_code_2"] = xr.where(
        ds_climate["climate_code"] == 2, 1, ds_climate["climate_code_2"]
    )
    ds_climate["climate_code_3"] = xr.where(
        ds_climate["climate_code"] == 3, 1, ds_climate["climate_code_3"]
    )

    # Drop the original 'climate_code' variable
    ds_climate = ds_climate.drop_vars("climate_code")

    da_post= da_post.drop_vars("time")
    dnbr = dnbr.drop_vars("time")
    dndvi = dndvi.drop_vars("time")
    dndmi = dndmi.drop_vars("time")
    dvari = dvari.drop_vars("time")

    # Merge all the datasets into a single result dataset
    result = xr.merge(
        [da_post, da_base, ds_lc, dnbr, dndvi, dvari, dndmi, ds_climate],
        compat="override",
    )

    return result


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
    "--task-id",
    "-t",
    type=str,
    default=None,
    help="REQUIRED. Burn Cube task id, e.g. Dec-21.",
)
@click.option(
    "--region-id",
    "-r",
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
def vic_rf_processing(
    task_id,
    region_id,
    process_cfg_url,
    overwrite,
):
    """
    Simple program to use VIC RF solution (Note: retrain the model by DEA) to generate VIC RF result.
    """

    logging_setup()

    dc = datacube.Datacube(
        app=f"Burn Cube K8s processing - {region_id}",
        config={
            "db_hostname": os.getenv("ODC_DB_HOSTNAME"),
            "db_password": os.getenv("ODC_DB_PASSWORD"),
            "db_username": os.getenv("ODC_DB_USERNAME"),
            "db_port": 5432,
            "db_database": os.getenv("ODC_DB_DATABASE"),
        },
    )
    hnrs_dc = datacube.Datacube(
        app=f"Burn Cube K8s processing - {region_id}",
        config={
            "db_hostname": os.getenv("HNRS_DB_HOSTNAME"),
            "db_password": os.getenv("HNRS_DC_DB_PASSWORD"),
            "db_username": os.getenv("HNRS_DC_DB_USERNAME"),
            "db_port": 5432,
            "db_database": os.getenv("HNRS_DC_DB_DATABASE"),
        },
    )
    # need to set the AWS login so that we can access the data we need
    os.environ["AWS_NO_SIGN_REQUEST"] = "Yes"

    process_cfg = helper.load_yaml_remote(process_cfg_url)

    pre_fire_gm_product_name = process_cfg["input_products"]["geomed_name"]
    feature_list = process_cfg["model_features"]
    output_folder = process_cfg["output_folder"]
    output_product_name = process_cfg["product"]["name"]
    task_table = process_cfg["task_table"]
    gm_product = process_cfg["input_products"]["geomed_name"]
    post_geomed_name = process_cfg["input_products"]["post_geomed_name"]
    wo_product = process_cfg["input_products"]["wofs_summary_name"]
    gm_measurements = process_cfg["input_products"]["input_gm_bands"]

    result_dict = task.task_to_ranges(task_id, task_table)

    time_pre = (result_dict["Period Start"], result_dict["Period End"])
    time_post = (result_dict["Mapping Period Start"], result_dict["Mapping Period End"])

    # Convert dictionary to a sorted string representation to ensure consistent hash
    dict_string = str(sorted(process_cfg.items()))

    # e.g., "https://dea-public-data-dev.s3.ap-southeast-2.amazonaws.com/projects/burn_cube/configs/"
    # + "RF_model_21_tiles_1000m_grid_3000m_to_7000m_buffer.joblib"
    model_url = process_cfg["model_path"]

    print(rioxarray.__version__)

    box = _get_gpgon(region_id)
    pgon = box[0]  # it always only one polygon there

    # Define the name of the Koppen climate GeoTIFF file
    geotiff_fname = "remapped_koppen_data_3classes_3577.tif"

    # auto download Koppen climate from AWS S3
    cfg_folder = "https://dea-public-data-dev.s3.ap-southeast-2.amazonaws.com/projects/burn_cube/configs/"

    # URL of the public S3 object
    url = cfg_folder + geotiff_fname

    download_file_from_s3_public(url, geotiff_fname)

    # Open the GeoTIFF file using xarray's open_rasterio function.
    climate_dataset = xr.open_rasterio(geotiff_fname)

    # Convert the opened raster data into an xarray dataset, where each band becomes a variable.
    climate_dataset = climate_dataset.to_dataset("band")

    # Rename variable 1 to 'climate_code' for clarity and easier access.
    climate_dataset = climate_dataset.rename({1: "climate_code"})

    climate_dataset = climate_dataset.where(
        climate_dataset["climate_code"] != 2147483647
    )

    # Generate a hash key using SHA-256
    hash_key = hashlib.sha256(dict_string.encode()).hexdigest()

    # Keep only the last 4 digits of the hash
    last_4_digits = hash_key[-4:]

    # Define the path to the saved machine learning model file.
    model_path = (
        f"RF_model_21_tiles_1000m_grid_3000m_to_7000m_buffer-{last_4_digits}.joblib"
    )

    # auto download Machine Learning model from AWS S3

    download_file_from_s3_public(model_url, model_path)

    # Load the machine learning model from the specified file using the `load` function from the `joblib` library.
    model = load(model_path)

    # Define the resolution of the geospatial data.
    resolution = (-30, 30)

    # Define the output coordinate reference system (CRS).
    output_crs = "epsg:3577"

    # Create a dictionary query object to pass to the `feature_layers` function
    query = {
        "resolution": resolution,
        "output_crs": output_crs,
        "measurements": gm_measurements,
        "geopolygon": pgon,
    }

    data = feature_layers(
        query,
        hnrs_dc,
        dc,
        time_pre,
        time_post,
        climate_dataset,
        pre_fire_gm_product_name,
        post_geomed_name,
        region_id
    ).squeeze()

    logger.info("Finish data loading")

    # this can make sure no issue on feature name order
    reorder_data = data[feature_list]

    predicted = predict_xr(
        model, reorder_data, proba=True, persist=True, clean=True, return_input=True
    ).compute()

    x_range = pgon.boundingbox.range_x
    y_range = pgon.boundingbox.range_y

    logger.info("Finish prediction")

    # Load the water observations data over the processed tile and analysis year
    wo = dc.load(
        product=wo_product,
        crs="EPSG:3577",
        output_crs="EPSG:3577",
        x=x_range,
        y=y_range,
        time=time_post[0].split("-")[0],
    )

    # Create water mask to mask pixels that have more than 20% wet observations
    # Plot the water mask
    wo_mask = wo.frequency > 0.2

    predicted_wofs = xr.where(wo_mask == 0, predicted, 0)

    logger.info("Apply WO Summary masking")

    # Define the size of the disk structuring element, measured in number of pixels.
    # The default value is 2.
    disk_size = 2

    # Remove the time index from the xr dataarray
    all_burn = predicted_wofs.Predictions.isel(time=0)

    # Perform an opening morphological operation on the `all_burn` dataarray
    opened_data = xr.DataArray(
        morphology.binary_opening(all_burn, morphology.disk(disk_size)),
        coords=all_burn.coords,
    )
    # Perform a closing morphological operation on the `opened_data` dataarray
    dilated_data = xr.DataArray(
        ndimage.binary_dilation(opened_data, morphology.disk(disk_size + 1)),
        coords=all_burn.coords,
    )

    # Set the post-processed data to the `all_burn_cleaned` variable, and convert to a float dtype
    all_burn_cleaned = dilated_data
    all_burn_cleaned = all_burn_cleaned.astype(int)
    all_burn_cleaned = all_burn_cleaned.astype("float64")

    # Reapply the wo mask, to remove burnt pixels over water bodies that the above closing created
    all_burn_cleaned = xr.where(wo_mask == 0, all_burn_cleaned, 0)

    # Ensure the crs attribute is set to 3577 using the wo dc
    all_burn_cleaned.attrs["crs"] = wo.crs

    nm_xy = region_id  # dynamic build from data loading process
    nm_date = time_post[0]  # get year information

    pred_tif = output_product_name + f"_{nm_xy}_{nm_date}_pred.tif"

    # find a way to remove the time dim
    # all_burn_cleaned = all_burn_cleaned[0].squeeze(dim='time')
    all_burn_ds = all_burn_cleaned.to_dataset(name="all_burn_ds")
    all_burn_ds = all_burn_ds.isel(time=0, drop=True)
    all_burn_da = all_burn_ds["all_burn_ds"]

    write_cog(geo_im=all_burn_da, fname=pred_tif, overwrite=True, nodata=-999)

    logger.info("Save result as: " + str(pred_tif))

    s3_file_uri = f"{output_folder}/{output_product_name}/3-0-0/{region_id[:3]}/{region_id[3:]}/{pred_tif}"

    logger.info("Upload result to AWS S3 file: " + str(s3_file_uri))

    # activate AWS credential from attached service account
    helper.get_and_set_aws_credentials()

    bc_io.upload_object_to_s3(pred_tif, s3_file_uri)

    logger.info("finish predication: " + str(region_id))

    # generate tif to segmentation
    tif_to_seg = output_product_name + f"_{nm_xy}_{nm_date}_seg.tif"

    write_cog(geo_im=data.dNDVI, fname=tif_to_seg, overwrite=True, nodata=-999)

    logger.info("Save result as: " + str(tif_to_seg))

    s3_file_uri = f"{output_folder}/{output_product_name}/3-0-0/{region_id[:3]}/{region_id[3:]}/{tif_to_seg}"

    logger.info("Upload result to AWS S3 file: " + str(s3_file_uri))

    bc_io.upload_object_to_s3(tif_to_seg, s3_file_uri)

    logger.info("finish segement: " + str(region_id))

    # Convert our mean NDVI xarray into a numpy array
    dndvi = rioxarray.open_rasterio(tif_to_seg).squeeze().values

    # Calculate the segments
    segments = quickshift(
        dndvi, kernel_size=2, convert2lab=False, max_dist=6, ratio=1.0
    )

    pred = rioxarray.open_rasterio(pred_tif).squeeze().drop_vars("band")

    count, _sum = _stats(pred, labels=segments, index=segments)
    mode = _sum > (count / 2)

    # Expand the mask dimensions to match the DataArray
    # expanded_mode = np.expand_dims(mode, axis=0)

    mode = xr.DataArray(
        mode, coords=pred.coords, dims=pred.dims, attrs=pred.attrs
    ).astype(np.int16)

    pred_object_tif = output_product_name + f"_{nm_xy}_{nm_date}_prediction_object.tif"

    write_cog(mode, pred_object_tif, overwrite=True)

    logger.info("Save result as: " + str(pred_object_tif))

    s3_file_uri = f"{output_folder}/{output_product_name}/3-0-0/{region_id[:3]}/{region_id[3:]}/{pred_object_tif}"

    logger.info("Upload result to AWS S3 file: " + str(s3_file_uri))

    bc_io.upload_object_to_s3(pred_object_tif, s3_file_uri)

    logger.info("finish proection object filter: " + str(region_id))


if __name__ == "__main__":
    vic_rf_processing()
