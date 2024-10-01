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
def rbr_processing(
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
    output_folder = process_cfg["output_folder"]
    time_pre = ("2017-01-01", "2017-12-31")
    feature_list = process_cfg["model_features"]

    output_product_name = process_cfg["product"]["name"]

    box = _get_gpgon(region_id)
    pgon = box[0]  # it always only one polygon there

    # Define the resolution of the geospatial data.
    resolution = (-30, 30)

    # Define the output coordinate reference system (CRS).
    output_crs = "epsg:3577"

    # PRE FIRE DATA
    # load in the 4 (financial or calendar) year geomedian
    ds = hnrs_dc.load(product="ga_ls8c_nbart_gm_4cyear_3", geopolygon=pgon,
                    time=("2017-01-01", "2017-12-31"), output_crs=output_crs)

    # POST FIRE DATA
    #load the post fire data, or the year of interest
    post_ds = load_ard(dc= dc, 
                geopolygon = pgon,
                time=("2020-01-01", "2020-12-31")
                group_by='solar_day',
                min_gooddata=0.7,
                output_crs=output_crs)

    #load wo to mask out the ocean later
    wofs_summary = dc.load(product="ga_ls_wo_fq_cyear_3",
                geopolygon = pgon,
                time=("2020")) #calendar year
    
    # normalised burn ratio
    pre_nbr = (ds.nir - ds.swir2) / (ds.nir + ds.swir2)

    # normalised burn ratio
    post_nbr = (post_ds.nbart_nir - post_ds.nbart_swir_2) / (post_ds.nbart_nir + post_ds.nbart_swir_2)

    # delta normalised burn ratio
    delta_nbr = pre_nbr.squeeze("time")-post_nbr

    RBR = delta_nbr / (pre_nbr.squeeze("time") + 1.001) #RBR

    #masking the water and ocean
    wofs_summary_frequency = wofs_summary.frequency

    # Create a water mask by identifying areas with water frequency greater than or equal to 0.2
    #water_mask = xr.where(wofs_summary_frequency < 0.2, 1., wofs_summary_frequency*0.)
    #water_mask.plot()
    # NEW
    # # Create a water mask by identifying areas with water frequency greater than or equal to 0.2
    water_mask = wofs_summary_frequency > 0.2
    water_mask = water_mask.squeeze("time")
    # water_mask.plot()

    # mask the delta normalised burn ratio
    #wo_delta_nbr = water_mask.squeeze("time") * delta_nbr

    wo_RBR = xr.where(water_mask == 0, RBR, -1)
    #wo_delta_nbr.plot(col="time", col_wrap=2, vmin=-1, vmax=1, cmap="PiYG")

    # finding the most burnt characteristic for each pixel in each dataset for the time period
    RBR_reduced = wo_RBR.max("time") 

    threshold_RBR = (RBR_reduced >= 0.3 )*1 #RBR paper 2014

    threshold_RBR.attrs["crs"] = wofs_summary.crs
    threshold_RBR = threshold_RBR.astype('float64')

    pred_tif = output_product_name + f"_{nm_xy}_{nm_date}_rbr_pred.tif"

    write_cog(geo_im=threshold_RBR, fname=pred_tif, overwrite=True, nodata=-999)

    logger.info("Save result as: " + str(pred_tif))

    s3_file_uri = f"{output_folder}/{output_product_name}/3-0-0/{region_id[:3]}/{region_id[3:]}/{pred_tif}"

    logger.info("Upload result to AWS S3 file: " + str(s3_file_uri))

    # activate AWS credential from attached service account
    helper.get_and_set_aws_credentials()

    bc_io.upload_object_to_s3(pred_tif, s3_file_uri)

    logger.info("finish predication: " + str(region_id))


if __name__ == "__main__":
    vic_rf_processing()
