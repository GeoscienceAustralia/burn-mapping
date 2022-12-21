"""CLI: Run Burn Cube analysis based on task_id and region_id
Geoscience Australia
2022
"""

import os

import boto3
import click
import datacube
import dea_tools.datahandling
import geopandas as gpd

import dea_burn_cube.__version__
import dea_burn_cube.utils as utils


def gqa_predicate(ds):
    return ds.metadata.gqa_iterative_mean_xy <= 1


def get_geomed_ds(region_id, split_count, period, hnrs_config, geomed_bands):
    hnrs_dc = datacube.Datacube(app="geomed_loading", config=hnrs_config)

    metadata_files = hnrs_dc.find_datasets(product="ga_ls8c_nbart_gm_4cyear_3")
    target_dataset = [
        e
        for e in metadata_files
        if e.metadata_doc["properties"]["odc:region_code"] == region_id
    ][0]

    metadata = hnrs_dc.index.datasets.get(str(target_dataset.id))

    geometry_list = [metadata.extent]

    polygon = gpd.GeoDataFrame(
        index=range(len(geometry_list)), crs="epsg:3577", geometry=geometry_list
    )

    geomed = hnrs_dc.load(
        datasets=[metadata],
        output_crs="EPSG:3577",
        resolution=(-30, 30),
        measurments=geomed_bands,
        dask_chunks={},
    )

    geomed = geomed[geomed_bands].to_array(dim="band").to_dataset(name="geomedian")

    return polygon, geomed


def generate_subregion_result(
    dc,
    geomed,
    ard_bands,
    period,
    mappingperiod,
    polgyon_geometry,
    x_i,
    y_i,
    split_count,
    x_code,
    y_code,
):

    gpgon = datacube.utils.geometry.Geometry(polgyon_geometry, crs="epsg:3577")

    ard = dea_tools.datahandling.load_ard(
        dc,
        products=["ga_ls8c_ard_3"],
        measurements=ard_bands,
        geopolygon=gpgon,
        output_crs="EPSG:3577",
        resolution=(-30, 30),
        resampling={"fmask": "nearest", "*": "bilinear"},
        dask_chunks={},
        predicate=gqa_predicate,
        time=period,
        group_by="solar_day",
    )

    ard = ard[ard_bands].to_array(dim="band").to_dataset(name="ard")

    interval = int(len(ard.ard.x) / split_count)

    ard = ard.ard.isel(
        x=range(x_i * interval, (x_i + 1) * interval),
        y=range(y_i * interval, (y_i + 1) * interval),
    )
    geomed = geomed.geomedian.isel(
        x=range(x_i * interval, (x_i + 1) * interval),
        y=range(y_i * interval, (y_i + 1) * interval),
    )

    geomed = geomed.load()
    ard = ard.load()

    dis = utils.distances(ard, geomed)
    outliers_result = utils.outliers(ard, dis)

    del ard, dis

    mapping_ard = dea_tools.datahandling.load_ard(
        dc,
        products=["ga_ls8c_ard_3"],
        measurements=ard_bands,
        geopolygon=gpgon,
        output_crs="EPSG:3577",
        resolution=(-30, 30),
        resampling={"fmask": "nearest", "*": "bilinear"},
        dask_chunks={},
        predicate=gqa_predicate,
        time=mappingperiod,
        group_by="solar_day",
    )

    mapping_ard = mapping_ard[ard_bands].to_array(dim="band").to_dataset(name="ard")
    mapping_ard = mapping_ard.ard.isel(
        x=range(x_i * interval, (x_i + 1) * interval),
        y=range(y_i * interval, (y_i + 1) * interval),
    )

    mapping_ard = mapping_ard.load()

    mapping_dis = utils.distances(mapping_ard, geomed)

    return utils.severitymapping(
        mapping_dis, outliers_result, mappingperiod, method="NBR", growing=True
    )


@click.group()
@click.version_option(version=dea_burn_cube.__version__)
def main():
    """Run dea-burn-cube."""


@main.command(no_args_is_help=True)
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
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    # Don't mandate existence since this might be s3://.
    help="REQUIRED. Path to the output directory.",
)
def burn_cube_run(
    task_id,
    region_id,
    output,
):
    bc_running_task = utils.generate_task(task_id)

    geomed_bands = ["red", "green", "blue", "nir", "swir1", "swir2"]

    ard_bands = [
        f"nbart_{band}" for band in ("red", "green", "blue", "nir", "swir_1", "swir_2")
    ]

    period = (bc_running_task["Period Start"], bc_running_task["Period End"])
    mappingperiod = (
        bc_running_task["Mapping Period Start"],
        bc_running_task["Mapping Period End"],
    )

    # The following variables passed by K8s Pod manifest
    hnrs_config = {
        "db_hostname": os.getenv("HNRS_DB_HOSTNAME"),
        "db_password": os.getenv("HNRS_DC_DB_PASSWORD"),
        "db_username": os.getenv("HNRS_DC_DB_USERNAME"),
        "db_port": 5432,
        "db_database": os.getenv("HNRS_DC_DB_DATABASE"),
    }

    odc_config = {
        "db_hostname": os.getenv("ODC_DB_HOSTNAME"),
        "db_password": os.getenv("ODC_DB_PASSWORD"),
        "db_username": os.getenv("ODC_DB_USERNAME"),
        "db_port": 5432,
        "db_database": os.getenv("ODC_DB_DATABASE"),
    }

    dc = datacube.Datacube(app="Burn Cube K8s processing", config=odc_config)

    split_count = 4

    x_code = region_id[:3]
    y_code = region_id[3:]

    polygon, geomed = get_geomed_ds(
        region_id, split_count, period, hnrs_config, geomed_bands
    )

    for x_i in range(split_count):
        for y_i in range(split_count):

            out = generate_subregion_result(
                dc,
                geomed,
                ard_bands,
                period,
                mappingperiod,
                polygon.geometry[0],
                x_i,
                y_i,
                split_count,
                x_code,
                y_code,
            )

            s3_file_path = f"{task_id}/{region_id}/BurnMapping-{task_id}-{region_id}-{x_i}-{y_i}.nc"
            local_file_path = f"BurnMapping-{task_id}-{region_id}-{x_i}-{y_i}.nc"

            from urllib.parse import urlparse

            o = urlparse(output)

            target_file_path = f"{o.path}/{s3_file_path}"

            # let us assume the output is an AWS S3 path
            if out:

                comp = dict(zlib=True, complevel=5)
                encoding = {var: comp for var in out.data_vars}  # compression

                # this will save it in the current working directory
                out.to_netcdf(local_file_path, encoding=encoding)

                s3 = boto3.client("s3")

                with open(local_file_path, "rb") as f:
                    s3.upload_fileobj(f, o.netloc, target_file_path[1:])


if __name__ == "__main__":
    main()
