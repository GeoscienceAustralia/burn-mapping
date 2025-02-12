import logging
import os
import re
import sys
from typing import Tuple

import click
import datacube
import xarray as xr
from datacube.utils.cog import write_cog
from dea_tools.datahandling import load_ard
from odc.dscache.tools.tiling import parse_gridspec_with_name

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


def dilrode_Delta_dataset(burn_dataset: xr.Dataset)-> xr.DataArray:
    dilated_data = xr.DataArray(morphology.binary_closing(burn_dataset, morphology.disk(3)).astype(burn_dataset.dtype),
                                 coords=burn_dataset.coords)
    erroded_data = xr.DataArray(morphology.erosion(dilated_data, morphology.disk(3)).astype(burn_dataset.dtype),
                                 coords=burn_dataset.coords)
    dilated_data = xr.DataArray(ndimage.binary_dilation(erroded_data, morphology.disk(3)).astype(burn_dataset.dtype),
                                 coords=burn_dataset.coords)
    return dilated_data


def save_and_upload(geo_im, product_name, region_id, output_folder, output_product_name):

    """ Saves GeoTIFF and uploads to S3 """
    pred_tif = f"{output_product_name}_{region_id}_2020_cyear_{product_name}_pred.tif"

    write_cog(geo_im=geo_im, fname=pred_tif, overwrite=True, nodata=-999)
    logger.info(f"Save result as: {pred_tif}")

    s3_file_uri = f"{output_folder}/{output_product_name}/3-0-0/{region_id[:3]}/{region_id[3:]}/{pred_tif}"
    logger.info(f"Upload result to AWS S3 file: {s3_file_uri}")

    # Activate AWS credentials and upload
    helper.get_and_set_aws_credentials()
    bc_io.upload_object_to_s3(pred_tif, s3_file_uri)
    logger.info(f"Finished processing {product_name} for region {region_id}")


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

    # pre_fire_gm_product_name = process_cfg["input_products"]["geomed_name"]
    output_folder = process_cfg["output_folder"]
    # time_pre = ("2017-01-01", "2017-12-31")

    output_product_name = process_cfg["product"]["name"]

    box = _get_gpgon(region_id)
    pgon = box[0]  # it always only one polygon there

    # Define the resolution of the geospatial data.
    # resolution = (-30, 30)

    # Define the output coordinate reference system (CRS).
    output_crs = "epsg:3577"

    # PRE FIRE DATA
    # load in the 4 (financial or calendar) year geomedian
    ds = hnrs_dc.load(
        product="ga_ls8c_nbart_gm_4cyear_3",
        geopolygon=pgon,
        time=("2017-01-01", "2017-12-31"),
        output_crs=output_crs,
    )

    # POST FIRE DATA
    # load the post fire data, or the year of interest
    post_ds = load_ard(
        dc=dc,
        products=["ga_ls5t_ard_3", "ga_ls7e_ard_3", "ga_ls8c_ard_3"],
        geopolygon=pgon,
        time=("2020-01-01", "2020-12-31"),
        group_by="solar_day",
        min_gooddata=0.7,
        output_crs=output_crs,
    )

    # load wo to mask out the ocean later
    wofs_summary = dc.load(
        product="ga_ls_wo_fq_cyear_3", geopolygon=pgon, time=("2020")
    )  # calendar year

    # common index layers

    # bare soil index
    pre_bsi = (
        (ds.swir2 + ds.red)
        - (ds.nir + ds.blue)
    ) / (
        (ds.swir2 + ds.red)
        + (ds.nir + ds.blue)
    )

    # bare soil index
    post_bsi = (
        (post_ds.nbart_swir_2 + post_ds.nbart_red)
        - (post_ds.nbart_nir + post_ds.nbart_blue)
    ) / (
        (post_ds.nbart_swir_2 + post_ds.nbart_red)
        + (post_ds.nbart_nir + post_ds.nbart_blue)
    )

    # normalised difference vegetation index
    pre_ndvi = (ds.nir - ds.red) / (
        ds.nir + ds.red
    )

    # normalised difference vegetation index
    post_ndvi = (post_ds.nbart_nir - post_ds.nbart_red) / (
        post_ds.nbart_nir + post_ds.nbart_red
    )

    # normalised burn ratio
    pre_nbr = (ds.nir - ds.swir2) / (ds.nir + ds.swir2)

    # normalised burn ratio
    post_nbr = (post_ds.nbart_nir - post_ds.nbart_swir_2) / (
        post_ds.nbart_nir + post_ds.nbart_swir_2
    )

    # delta normalised burn ratio
    delta_nbr = pre_nbr.squeeze("time") - post_nbr

    #Wetness
    pre_tcw = (0.2578*ds.blue + 0.2305*ds.green + 0.0883*ds.red + 
            0.1071*ds.nir - 0.7611*ds.swir1 - 0.5308*ds.swir2)

    #Brightness
    pre_tcb = (0.3510*ds.blue + 0.3813*ds.green + 0.3437*ds.red + 
            0.7196*ds.nir+ 0.2396*ds.swir1 + 0.1949*ds.swir2)

    #Greenness
    pre_tcg = (-0.3599*ds.blue - 0.3533*ds.green - 0.4734*ds.red + 
            0.6633*ds.nir + 0.0087*ds.swir1 - 0.2856*ds.swir2)

    #Wetness
    post_tcw = (0.2578*post_ds.nbart_blue + 0.2305*post_ds.nbart_green + 0.0883*post_ds.nbart_red + 
            0.1071*post_ds.nbart_nir - 0.7611*post_ds.nbart_swir_1 - 0.5308*post_ds.nbart_swir_2)
    
    #Brightness
    post_tcb = (0.3510*post_ds.nbart_blue + 0.3813*post_ds.nbart_green + 0.3437*post_ds.nbart_red + 
            0.7196*post_ds.nbart_nir+ 0.2396*post_ds.nbart_swir_1 + 0.1949*post_ds.nbart_swir_2)

    #Greenness
    post_tcg = (-0.3599*post_ds.nbart_blue - 0.3533*post_ds.nbart_green - 0.4734*post_ds.nbart_red + 
            0.6633*post_ds.nbart_nir + 0.0087*post_ds.nbart_swir_1 - 0.2856*post_ds.nbart_swir_2)

    # masking the water and ocean
    wofs_summary_frequency = wofs_summary.frequency

    # # Create a water mask by identifying areas with water frequency greater than or equal to 0.2
    water_mask = wofs_summary_frequency > 0.2
    water_mask = water_mask.squeeze("time")

    # activate AWS credential from attached service account
    helper.get_and_set_aws_credentials()

    # 1. Single RBR prediction
    RBR = delta_nbr / (pre_nbr.squeeze("time") + 1.001)  # RBR

    wo_RBR = xr.where(water_mask == 0, RBR, -1)

    # finding the most burnt characteristic for each pixel in each dataset for the time period
    RBR_reduced = wo_RBR.max("time")

    threshold_RBR = (RBR_reduced >= 0.3) * 1  # RBR paper 2014

    threshold_RBR.attrs["crs"] = wofs_summary.crs
    threshold_RBR = threshold_RBR.astype("float64")

    save_and_upload(threshold_RBR, "single_rbr", region_id, output_folder, output_product_name)

    # 2. Single NBR
    wo_delta_nbr = xr.where(water_mask == 0, delta_nbr, -1)

    delta_nbr_reduced = wo_delta_nbr.max("time") 

    threshold_dnbr = (delta_nbr_reduced >= 0.44 ) * 1 #USGS #0.44

    threshold_dnbr.attrs["crs"] = wofs_summary.crs
    threshold_dnbr = threshold_dnbr.astype("float64")

    save_and_upload(threshold_dnbr, "single_nbr", region_id, output_folder, output_product_name)

    # 3. Single RdNBR

    RdNBR = delta_nbr/(abs(pre_nbr.squeeze("time")) ** 0.5)

    wo_RdNBR = xr.where(water_mask == 0, RdNBR, -1)

    RdNBR_reduced = wo_RdNBR.max("time")

    threshold_RdNBR = (RdNBR_reduced >= 0.33 )*1 #Szajewska 2018

    threshold_RdNBR.attrs["crs"] = wofs_summary.crs
    threshold_RdNBR = threshold_RdNBR.astype("float64")

    save_and_upload(threshold_RdNBR, "single_rdnbr", region_id, output_folder, output_product_name)

    # 4. Stacked NBR

    # delta normalised difference vegetation index
    delta_ndvi = pre_ndvi.squeeze("time")-post_ndvi
    # delta bare soil index
    delta_bsi = pre_bsi.squeeze("time")-post_bsi

    wo_delta_ndvi = xr.where(water_mask == 0, delta_ndvi, -1)
    wo_delta_bsi = xr.where(water_mask == 0, delta_bsi, 1)

    delta_ndvi_reduced = wo_delta_ndvi.max("time") 
    delta_bsi_reduced = wo_delta_bsi.min("time") 

    # # standardising so all on same negative to positive scale so that very burnt =1
    delta_bsi_reduced = delta_bsi_reduced *-1 # 

    # take the threshold of the various characteristics
    threshold_dbsi = (delta_bsi_reduced >= 0.55 )*1 #Nguyen 2021
    threshold_dnbr = (delta_nbr_reduced >= 0.44 )*1 #USGS #0.44
    threshold_dndvi = (delta_ndvi_reduced >= 0.65 )*1 #Szajewska 2018

    RdNBR_stacked_agreement = threshold_dbsi + threshold_dndvi + threshold_dnbr
    RdNBR_stacked_thresholded = RdNBR_stacked_agreement >= 2

    # only process stacked result?
    RdNBR_stacked_thresholded = dilrode_Delta_dataset(RdNBR_stacked_thresholded)

    save_and_upload(RdNBR_stacked_thresholded, "stacked_nbr", region_id, output_folder, output_product_name)

    # 5. Stacked RBR

    RBR_stacked_agreement = threshold_dbsi + threshold_dndvi + threshold_RBR
    RBR_stacked_thresholded = RBR_stacked_agreement >= 2
    
    # only process stacked result?
    RBR_stacked_thresholded = dilrode_Delta_dataset(RBR_stacked_thresholded)

    save_and_upload(RBR_stacked_thresholded, "stacked_rbr", region_id, output_folder, output_product_name)

    # 6. Stacked RdNBR

    RdNBR_stacked_agreement = threshold_dbsi + threshold_dndvi + threshold_RdNBR
    RdNBR_stacked_thresholded = RdNBR_stacked_agreement >= 2

    # only process stacked result?
    RdNBR_stacked_thresholded = dilrode_Delta_dataset(RdNBR_stacked_thresholded)

    save_and_upload(RdNBR_stacked_thresholded, "stacked_rdnbr", region_id, output_folder, output_product_name)

    # 7. stacked dDI

    delta_tcw = pre_tcw.squeeze("time")-post_tcw
    delta_tcb = pre_tcb.squeeze("time")-post_tcb
    delta_tcg = pre_tcg.squeeze("time")-post_tcg

    delta_DI = ((delta_tcg + delta_tcw -0.5*delta_tcb)/10000)

    # mask the delta normalised burn ratio
    wo_dDI = xr.where(water_mask == 0, delta_DI, -1)

    # finding the most burnt characteristic for each pixel in each dataset for the time period
    delta_nbr_reduced = wo_delta_nbr.max("time") 
    dDI_reduced = wo_dDI.max("time") 
    RBR_reduced = wo_RBR.max("time") 

    delta_ndvi_reduced = wo_delta_ndvi.max("time") 
    delta_bsi_reduced = wo_delta_bsi.min("time") 

    # # standardising so all on same negative to positive scale so that very burnt =1
    delta_bsi_reduced = delta_bsi_reduced *-1 # 

    # take the threshold of the various characteristics
    threshold_dbsi = (delta_bsi_reduced >= 0.55 )*1 #Nguyen 2021
    threshold_dnbr = (delta_nbr_reduced >= 0.44 )*1 #USGS #0.44
    threshold_dndvi = (delta_ndvi_reduced >= 0.65 )*1 #Szajewska 2018

    threshold_RBR = (RBR_reduced >= 0.3 )*1 #Nguyen 2021
    #threshold_dnbr = (delta_nbr_reduced >= 0.44 )*1 #USGS #0.44
    threshold_dDI = (dDI_reduced >= 0.3 )*1 #Szajewska 2018

    dDI_stacked_agreement = threshold_dbsi + threshold_dndvi + threshold_dDI
    dDI_stacked_thresholded = dDI_stacked_agreement >= 2 

    # only process stacked result?
    dDI_stacked_thresholded = dilrode_Delta_dataset(dDI_stacked_thresholded)

    save_and_upload(dDI_stacked_thresholded, "stacked_dDI", region_id, output_folder, output_product_name)

    # 8. single dDI
    save_and_upload(threshold_dDI, "single_dDI", region_id, output_folder, output_product_name)

    # 9. stacked dDI RBR
    dDI_RBR_stacked_agreement = threshold_dbsi + threshold_dndvi + threshold_RBR + threshold_dDI
    dDI_RBR_stacked_thresholded = dDI_RBR_stacked_agreement >= 2 

    # only process stacked result?
    dDI_RBR_stacked_thresholded = dilrode_Delta_dataset(dDI_RBR_stacked_thresholded)

    save_and_upload(dDI_RBR_stacked_thresholded, "stacked_dDI_RBR", region_id, output_folder, output_product_name)


if __name__ == "__main__":
    rbr_processing()
