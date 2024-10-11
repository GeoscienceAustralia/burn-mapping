import rioxarray
import s3fs
import xarray as xr
from datacube.utils import geometry
from datacube.utils.cog import write_cog

from dea_burn_cube import bc_io, helper

# Initialize the S3 filesystem (using anonymous access)
fs = s3fs.S3FileSystem(anon=True)

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


def process_files(match_products, region_id, output_folder):
    """
    Search for corresponding files from the match_products based on region id
    """

    pair_files = []

    # Collect matching files from each product
    for match_product in match_products:
        # Build the target folder path dynamically
        # Let us assume all NBIC product under the same project folder:
        # s3://dea-public-data-dev/projects/burn_cube/derivative
        target_folder = (
            f"{output_folder/match_product['product_name']}/3-0-0/{region_id}/"
        )

        # List files in the specific target folder
        all_files = fs.glob(target_folder + "**")

        # Filter matching files by extension and store with product weight
        matching_files = [
            file for file in all_files if file.endswith(match_product["extension_name"])
        ]

        if matching_files:
            pair_files.append(
                {
                    "file_path": matching_files[0],
                    "product_weight": match_product["product_weight"],
                }
            )

    # If no files matched, skip further processing
    if not pair_files:
        return None

    # Open and process all matching files
    da_list = []
    for pair_file in pair_files:
        # Open raster data from S3
        da = rioxarray.open_rasterio(f"s3://{pair_file['file_path']}")
        # Apply product weight
        da_list.append(da * pair_file["product_weight"])

    # Concatenate along a new "variable" dimension and sum across it
    combined = xr.concat(da_list, dim="variable")
    sum_summary = combined.sum(dim="variable")

    # Add CRS (Coordinate Reference System) to the output
    sum_summary.attrs["crs"] = geometry.CRS("EPSG:3577")

    return sum_summary


@click.command(no_args_is_help=True)
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
def stacking_processing(
    region_id,
    process_cfg_url,
    overwrite,
):
    """
    Simple program to load all sub-products (e.g., burn cube, DEA RF and DEA RBR) to get
    Stacking result as GeoTIFF file.
    """

    logging_setup()

    process_cfg = helper.load_yaml_remote(process_cfg_url)

    match_products = process_cfg["match_products"]
    output_folder = process_cfg["output_folder"]

    output_product_name = process_cfg["product"]["name"]

    # activate AWS credential from attached service account
    helper.get_and_set_aws_credentials()

    # Limit processing to the first 4 files for efficiency (or batch processing)
    result = process_files(match_products, region_id, output_folder)

    if result:
        sum_summary, pattern = result
        # Write the result to a COG file

        pred_tif = ("dea_nbic_stacking_" + pattern.replace("/", "") + "_2020.tif",)

        write_cog(geo_im=sum_summary, fname=pred_tif, overwrite=True, nodata=-999)

        logger.info("Save result as: " + str(pred_tif))

        s3_file_uri = f"{output_folder}/{output_product_name}/3-0-0/{region_id[:3]}/{region_id[3:]}/{pred_tif}"

        bc_io.upload_object_to_s3(pred_tif, s3_file_uri)


if __name__ == "__main__":
    stacking_processing()
