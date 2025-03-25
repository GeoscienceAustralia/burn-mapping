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
from scipy.ndimage import binary_dilation

# Import missing dependencies for morphology operations
from skimage import morphology

from dea_burn_cube import bc_io, helper, task

# Set logging level for botocore and basic logging configuration.
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


def get_geometry_and_geobox(
    region_id: str,
) -> Tuple[datacube.utils.geometry.Geometry, datacube.utils.geometry._base.GeoBox]:
    """
    Return a geometry and corresponding geobox for the given region_id.

    The region_id should be in the format "x<number>y<number>" (e.g., "x30y29").
    """
    _, gridspec = parse_gridspec_with_name("au-30")
    pattern = r"x(\d+)y(\d+)"
    match = re.match(pattern, region_id)
    if not match:
        raise ValueError(f"Invalid region_id format: {region_id}")
    x, y = int(match.group(1)), int(match.group(2))
    geobox = gridspec.tile_geobox((x, y))
    geometry = datacube.utils.geometry.Geometry(geobox.extent.geom, crs="epsg:3577")
    return geometry, geobox


def apply_morphological_operations(
    data: xr.DataArray, disk_size: int = 3
) -> xr.DataArray:
    """
    Apply a series of morphological operations (closing, erosion, and dilation)
    to clean up the binary thresholded result.
    """
    # Binary closing to fill small holes
    closed = morphology.binary_closing(data, morphology.disk(disk_size)).astype(
        data.dtype
    )
    # Erode to remove small spurious pixels
    eroded = morphology.erosion(closed, morphology.disk(disk_size)).astype(data.dtype)
    # Dilate to smooth the edges again
    dilated = binary_dilation(eroded, structure=morphology.disk(disk_size)).astype(
        data.dtype
    )
    return xr.DataArray(dilated, coords=data.coords)


def prepare_dataarray(
    da: xr.DataArray, crs: str, dtype: str = "float64"
) -> xr.DataArray:
    """
    Set the CRS attribute and cast the DataArray to the desired dtype.
    """
    da.attrs["crs"] = crs
    return da.astype(dtype)


def save_and_upload(
    geo_im: xr.DataArray,
    product_name: str,
    region_id: str,
    output_folder: str,
    output_product_name: str,
) -> None:
    """
    Save the GeoTIFF using write_cog and upload the file to S3.
    """

    pred_tif = f"{output_product_name}_{region_id}_2020_cyear_{product_name}_pred.tif"
    write_cog(geo_im=geo_im, fname=pred_tif, overwrite=True, nodata=-999)
    logger.info(f"Saved result as: {pred_tif}")

    # Build the S3 URI (splitting region_id into two parts)
    s3_file_uri = f"{output_folder}/{output_product_name}/3-0-0/{region_id[:3]}/{region_id[3:]}/{pred_tif}"
    logger.info(f"Uploading result to AWS S3: {s3_file_uri}")

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
    help="REQUIRED. Region id in AU-30 Grid (format: x<number>y<number>).",
)
@click.option(
    "--process-cfg-url",
    "-p",
    type=str,
    default=None,
    help="REQUIRED. URL to Burn Cube process cfg YAML file.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Rerun scenes that have already been processed.",
)
def rbr_processing(
    task_id: str, region_id: str, process_cfg_url: str, overwrite: bool
) -> None:
    """
    Process burn cube data using various indices and upload results to S3.

    The script loads both pre-fire and post-fire data, calculates a set of indices,
    applies thresholds and morphological operations, and then saves and uploads each
    product (e.g. single RBR, single NBR, stacked indices, etc.).
    """
    logging_setup()

    # Create two datacube instances for different configurations
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

    # Allow anonymous AWS access if necessary
    os.environ["AWS_NO_SIGN_REQUEST"] = "Yes"

    process_cfg = helper.load_yaml_remote(process_cfg_url)
    output_folder = process_cfg["output_folder"]
    output_product_name = process_cfg["product"]["name"]
    task_table = process_cfg["task_table"]
    gm_product = process_cfg["input_products"]["geomed_name"]
    wo_product = process_cfg["input_products"]["wofs_summary_name"]

    result_dict = task.task_to_ranges(task_id, task_table)

    # --- Early Check: Verify if a representative output already exists ---
    # Here we check for the "single_rbr" product file. If it exists and overwrite is disabled,
    # we exit early to save processing costs.
    rep_product = "single_rbr"
    rep_pred_tif = f"{output_product_name}_{region_id}_2020_cyear_{rep_product}_pred.tif"
    rep_s3_uri = f"{output_folder}/{output_product_name}/3-0-0/{region_id[:3]}/{region_id[3:]}/{rep_pred_tif}"
    if not overwrite and helper.check_s3_file_exists(rep_s3_uri):
        logger.info(f"Representative output already exists in S3: {rep_s3_uri}. Exiting as overwrite is disabled.")
        sys.exit(0)

    pgon, _ = get_geometry_and_geobox(region_id)
    output_crs = "epsg:3577"

    geomed_datasets = hnrs_dc.find_datasets(
        product=gm_product,
        geopolygon=pgon,
        time=(result_dict["Period Start"], result_dict["Period End"]),
    )

    if len(geomed_datasets) == 0:
        logger.info(f"Cannot find 4 Year GM dataset: {region_id}")
        sys.exit(0)

    # Load pre-fire (4-cycle geomedian) and post-fire data
    ds = hnrs_dc.load(
        product=gm_product,
        geopolygon=pgon,
        time=(result_dict["Period Start"], result_dict["Period End"]),
        output_crs=output_crs,
    )

    post_ds = load_ard(
        dc=dc,
        products=["ga_ls5t_ard_3", "ga_ls7e_ard_3", "ga_ls8c_ard_3"],
        cloud_mask="fmask",
        geopolygon=pgon,
        time=(result_dict["Mapping Period Start"], result_dict["Mapping Period End"]),
        group_by="solar_day",
        # min_gooddata=0.7,
        output_crs=output_crs,
        dataset_maturity="final",
        gqa_iterative_mean_xy=[-1, 1],
        skip_broken_datasets=True,
    )

    # Load water frequency summary and create a water mask.
    wofs_summary = dc.load(
        product=wo_product,
        geopolygon=pgon,
        time=result_dict["Mapping Period Start"].split("-")[0],
    )
    
    water_mask = (
        wofs_summary.frequency.squeeze("time") > 0.2
    )  # areas with frequency > 0.2 are water

    helper.get_and_set_aws_credentials()  # ensure AWS credentials are set

    # Check if post_ds is None
    if post_ds is None:
        # Use a reference array for dimensions and coordinates; here we use wofs_summary.frequency as an example.
        ref_array = wofs_summary.frequency.squeeze("time")
        # Create an array with the same shape filled with np.nan
        nan_array = np.full(ref_array.shape, np.nan)
        # Convert to an xarray DataArray while preserving coordinate information
        dummy_da = xr.DataArray(nan_array, coords=ref_array.coords, dims=ref_array.dims)
        # Ensure that the dummy dataarray has the correct CRS using your helper function
        dummy_da = prepare_dataarray(dummy_da, wofs_summary.crs)

        logger.info(f"Cannot find any available ARD data at : {region_id}. Generate empty GeoTIFF placeholder files.")

        # List of all output product names for which we need placeholder files
        product_list = [
            "single_rbr",
            "single_nbr",
            "single_rdnbr",
            "stacked_nbr",
            "stacked_rbr",
            "stacked_rdnbr",
            "stacked_ddi",
            "single_ddi",
            "stacked_ddi_rbr",
        ]
        
        # Upload a placeholder file for each product
        for product_name in product_list:
            save_and_upload(dummy_da, product_name, region_id, output_folder, output_product_name)

        # stop processing
        sys.exit(0)


    # Compute common indices
    pre_bsi = ((ds.nbart_swir_2 + ds.nbart_red) - (ds.nbart_nir + ds.nbart_blue)) / (
        (ds.nbart_swir_2 + ds.nbart_red) + (ds.nbart_nir + ds.nbart_blue)
    )
    post_bsi = (
        (post_ds.nbart_swir_2 + post_ds.nbart_red)
        - (post_ds.nbart_nir + post_ds.nbart_blue)
    ) / (
        (post_ds.nbart_swir_2 + post_ds.nbart_red)
        + (post_ds.nbart_nir + post_ds.nbart_blue)
    )
    pre_ndvi = (ds.nbart_nir - ds.nbart_red) / (ds.nbart_nir + ds.nbart_red)
    post_ndvi = (post_ds.nbart_nir - post_ds.nbart_red) / (
        post_ds.nbart_nir + post_ds.nbart_red
    )
    pre_nbr = (ds.nbart_nir - ds.nbart_swir_2) / (ds.nbart_nir + ds.nbart_swir_2)
    post_nbr = (post_ds.nbart_nir - post_ds.nbart_swir_2) / (
        post_ds.nbart_nir + post_ds.nbart_swir_2
    )
    delta_nbr = pre_nbr.squeeze("time") - post_nbr

    # Additional indices for dDI calculation
    pre_tcw = (
        0.2578 * ds.nbart_blue
        + 0.2305 * ds.nbart_green
        + 0.0883 * ds.nbart_red
        + 0.1071 * ds.nbart_nir
        - 0.7611 * ds.nbart_swir_1
        - 0.5308 * ds.nbart_swir_2
    )
    pre_tcb = (
        0.3510 * ds.nbart_blue
        + 0.3813 * ds.nbart_green
        + 0.3437 * ds.nbart_red
        + 0.7196 * ds.nbart_nir
        + 0.2396 * ds.nbart_swir_1
        + 0.1949 * ds.nbart_swir_2
    )
    pre_tcg = (
        -0.3599 * ds.nbart_blue
        - 0.3533 * ds.nbart_green
        - 0.4734 * ds.nbart_red
        + 0.6633 * ds.nbart_nir
        + 0.0087 * ds.nbart_swir_1
        - 0.2856 * ds.nbart_swir_2
    )

    post_tcw = (
        0.2578 * post_ds.nbart_blue
        + 0.2305 * post_ds.nbart_green
        + 0.0883 * post_ds.nbart_red
        + 0.1071 * post_ds.nbart_nir
        - 0.7611 * post_ds.nbart_swir_1
        - 0.5308 * post_ds.nbart_swir_2
    )
    post_tcb = (
        0.3510 * post_ds.nbart_blue
        + 0.3813 * post_ds.nbart_green
        + 0.3437 * post_ds.nbart_red
        + 0.7196 * post_ds.nbart_nir
        + 0.2396 * post_ds.nbart_swir_1
        + 0.1949 * post_ds.nbart_swir_2
    )
    post_tcg = (
        -0.3599 * post_ds.nbart_blue
        - 0.3533 * post_ds.nbart_green
        - 0.4734 * post_ds.nbart_red
        + 0.6633 * post_ds.nbart_nir
        + 0.0087 * post_ds.nbart_swir_1
        - 0.2856 * post_ds.nbart_swir_2
    )

    # -----------------------
    # 1. Single RBR
    # -----------------------
    rbr = delta_nbr / (pre_nbr.squeeze("time") + 1.001)
    rbr_masked = xr.where(~water_mask, rbr, -1)
    rbr_reduced = rbr_masked.max("time")
    single_rbr = prepare_dataarray(
        (rbr_reduced >= 0.3).astype("float64"), wofs_summary.crs
    )
    save_and_upload(
        single_rbr, "single_rbr", region_id, output_folder, output_product_name
    )

    # -----------------------
    # 2. Single NBR
    # -----------------------
    nbr_masked = xr.where(~water_mask, delta_nbr, -1)
    nbr_reduced = nbr_masked.max("time")
    single_nbr = prepare_dataarray(
        (nbr_reduced >= 0.44).astype("float64"), wofs_summary.crs
    )
    save_and_upload(
        single_nbr, "single_nbr", region_id, output_folder, output_product_name
    )

    # -----------------------
    # 3. Single RdNBR
    # -----------------------
    rdnbr = delta_nbr / (abs(pre_nbr.squeeze("time")) ** 0.5)
    rdnbr_masked = xr.where(~water_mask, rdnbr, -1)
    rdnbr_reduced = rdnbr_masked.max("time")
    single_rdnbr = prepare_dataarray(
        (rdnbr_reduced >= 0.33).astype("float64"), wofs_summary.crs
    )
    save_and_upload(
        single_rdnbr, "single_rdnbr", region_id, output_folder, output_product_name
    )

    # -----------------------
    # 4. Stacked NBR
    # -----------------------
    delta_ndvi = pre_ndvi.squeeze("time") - post_ndvi
    delta_bsi = pre_bsi.squeeze("time") - post_bsi

    ndvi_masked = xr.where(~water_mask, delta_ndvi, -1)
    bsi_masked = xr.where(~water_mask, delta_bsi, 1)
    ndvi_reduced = ndvi_masked.max("time")
    # Invert bsi so that higher values mean more burnt
    bsi_reduced = (-bsi_masked).min("time")

    thresh_dbsi = (bsi_reduced >= 0.55).astype("float64")
    thresh_dnbr = (nbr_reduced >= 0.44).astype("float64")
    thresh_dndvi = (ndvi_reduced >= 0.65).astype("float64")

    stacked_nbr = prepare_dataarray(
        (thresh_dbsi + thresh_dndvi + thresh_dnbr >= 2), wofs_summary.crs
    )
    stacked_nbr = apply_morphological_operations(stacked_nbr)

    stacked_nbr = prepare_dataarray(stacked_nbr, wofs_summary.crs)

    save_and_upload(
        stacked_nbr, "stacked_nbr", region_id, output_folder, output_product_name
    )

    # -----------------------
    # 5. Stacked RBR
    # -----------------------
    thresh_rbr = (rbr_reduced >= 0.3).astype("float64")
    stacked_rbr = prepare_dataarray(
        (thresh_dbsi + thresh_dndvi + thresh_rbr >= 2), wofs_summary.crs
    )
    stacked_rbr = apply_morphological_operations(stacked_rbr)

    stacked_rbr = prepare_dataarray(stacked_rbr, wofs_summary.crs)

    save_and_upload(
        stacked_rbr, "stacked_rbr", region_id, output_folder, output_product_name
    )

    # -----------------------
    # 6. Stacked RdNBR
    # -----------------------
    thresh_rdnbr = (rdnbr_reduced >= 0.33).astype("float64")
    stacked_rdnbr = prepare_dataarray(
        (thresh_dbsi + thresh_dndvi + thresh_rdnbr >= 2), wofs_summary.crs
    )
    stacked_rdnbr = apply_morphological_operations(stacked_rdnbr)

    stacked_rdnbr = prepare_dataarray(stacked_rdnbr, wofs_summary.crs)

    save_and_upload(
        stacked_rdnbr, "stacked_rdnbr", region_id, output_folder, output_product_name
    )

    # -----------------------
    # 7. Stacked dDI
    # -----------------------
    delta_tcw = pre_tcw.squeeze("time") - post_tcw
    delta_tcb = pre_tcb.squeeze("time") - post_tcb
    delta_tcg = pre_tcg.squeeze("time") - post_tcg
    delta_di = (delta_tcg + delta_tcw - 0.5 * delta_tcb) / 10000

    di_masked = xr.where(~water_mask, delta_di, -1)
    ddi_reduced = di_masked.max("time")
    thresh_ddi = (ddi_reduced >= 0.3).astype("float64")

    stacked_ddi = prepare_dataarray(
        (thresh_dbsi + thresh_dndvi + thresh_ddi >= 2), wofs_summary.crs
    )
    stacked_ddi = apply_morphological_operations(stacked_ddi)

    stacked_ddi = prepare_dataarray(stacked_ddi, wofs_summary.crs)

    save_and_upload(
        stacked_ddi, "stacked_ddi", region_id, output_folder, output_product_name
    )

    # -----------------------
    # 8. Single dDI
    # -----------------------
    single_ddi = prepare_dataarray(thresh_ddi, wofs_summary.crs)
    save_and_upload(
        single_ddi, "single_ddi", region_id, output_folder, output_product_name
    )

    # -----------------------
    # 9. Stacked dDI RBR
    # -----------------------
    stacked_ddi_rbr = prepare_dataarray(
        (thresh_dbsi + thresh_dndvi + thresh_rbr + thresh_ddi >= 2), wofs_summary.crs
    )
    stacked_ddi_rbr = apply_morphological_operations(stacked_ddi_rbr)

    stacked_ddi_rbr = prepare_dataarray(stacked_ddi_rbr, wofs_summary.crs)

    save_and_upload(
        stacked_ddi_rbr,
        "stacked_ddi_rbr",
        region_id,
        output_folder,
        output_product_name,
    )


if __name__ == "__main__":
    rbr_processing()
