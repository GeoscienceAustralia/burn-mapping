# Import libraries required for AWS download
import os

import boto3
import click

# Import libraries required for ARDC dataset extraction and validation modules
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import requests
import xarray as xr

from dea_burn_cube import helper

# Import tools and functions from ARDC_burnt_area_mapping_tools.py
from ARDC_burnt_area_mapping_tools import (
    ardc_year_calc,
    calculate_classification_metrics,
    extract_xy,
    get_tifs,
    raster_folder_bbox,
    read_shapes,
    validation_climate_analysis,
    validation_stats,
)

# Import tools and functions from DEA tools
from datacube.utils.cog import write_cog
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter


def upload_folder_to_s3(local_folder_path, bucket_name, s3_folder):
    """
    Uploads a local folder to an S3 bucket.

    Parameters:
    - local_folder_path: The local path to the folder r upload.
    - bucket_name: The name of the S3 bucket.
    - s3_folder: The folder path inside the S3 bucket.
    """
    s3_client = boto3.client("s3")

    # Walk through the local folder
    for subdir, dirs, files in os.walk(local_folder_path):
        for file in files:
            full_path = os.path.join(subdir, file)
            # Construct the full S3 path
            s3_path = os.path.join(
                s3_folder, os.path.relpath(full_path, local_folder_path)
            )
            print(f"Uploading {full_path} to s3://{bucket_name}/{s3_path}")
            # Upload the file
            s3_client.upload_file(full_path, bucket_name, s3_path)


# Define a style function for the visualisation of the results
def result_visuals(styler, study_site, vali_year):
    styler.set_caption(
        "Climate Class Validation Results for {}, {}".format(
            study_site.replace("_", " "), vali_year
        )
    )
    styler.format(
        {
            "Precision": lambda x: f"{x*100:.1f}%",
            "Recall": lambda x: f"{x*100:.1f}%",
            "Area_Percent": lambda x: f"{x*100:.1f}%",
        }
    )
    styler.background_gradient(
        subset=["Precision", "Recall"], vmin=0.3, vmax=1, cmap="RdYlGn"
    )
    return styler


def generate_result_by_study_site_folder(
    algo_name,
    save_folder,
    study_site,
    year_basis,
    extra_months,
    CoastLineShapeFile,
    suffix,
    inputType,
    colpac,
    ClimateZoneShapeFile,
    ClimateLegend_fname,
    vali_year,
):

    year = 2020

    polygon, poly_crs = raster_folder_bbox(save_folder)

    time_period = ardc_year_calc(year_basis, year, extra_months)

    # Create a gdf using the polygon object created from the extent of BC geotifs
    # Set its crs and transform to the ardc_gdf crs
    poly_gdf = gpd.GeoDataFrame(geometry=[polygon])
    poly_gdf = poly_gdf.set_crs(poly_crs)

    # Change crs of both gdf to 3577
    # ardc_gdf = ardc_gdf.to_crs(3577)
    poly_gdf = poly_gdf.to_crs(3577)

    geopackage_outpath = "{}_{}_{}_ARDC_polygons.gpkg".format(
        study_site, year, year_basis
    )

    ardc_gdf_fil = gpd.read_file(geopackage_outpath)
    FigTitle = f"{algo_name} Results for {study_site} {year}-{year_basis}"
    GroundTruthFile = geopackage_outpath
    GTF, CLSF, StateBndry = read_shapes(GroundTruthFile, CoastLineShapeFile)

    TifList = get_tifs(save_folder, suffix)

    TifC = []

    for i in TifList:
        xx, yy = extract_xy(i)
        TifC.append([xx, yy])
    TifCoords = str(TifC)[1:-1]

    print(study_site, algo_name, TifList)

    # Create single xarray out of combined True/False/Positive/Negative tiles
    CombineList = []
    TPTotal, FNTotal, FPTotal, TNTotal = 0, 0, 0, 0
    for i, item in enumerate(TifList):
        Comby, TP, FN, FP, TN = validation_stats(
            item, GTF, CLSF, StateBndry, inputType, colpac, graph_out=False
        )
        Comby.name = "var"
        CombineList.append(Comby)
        TPTotal += TP
        FNTotal += FN
        FPTotal += FP
        TNTotal += TN

    try:
        CombineArray = xr.merge(CombineList).to_array()
        print("CombineArray created by xarray merge")
    except xr.MergeError:
        CombineArray = xr.combine_by_coords(CombineList).to_array()
        print("CombineArray created by xarray combine_by_coords")

    metric_list = ["precision", "recall", "f1-score"]
    accuracy_metrics = calculate_classification_metrics(
        tp=TPTotal, tn=TNTotal, fp=FPTotal, fn=FNTotal, metrics=metric_list
    )
    print(study_site, algo_name, accuracy_metrics)

    # Precision = TP/(TP+FP)
    if TPTotal + FPTotal > 0 and TPTotal >= 0:
        Prec = round(100 * accuracy_metrics["precision"], 1)
        PrecStr = "Precision = " + str(Prec) + "%"
    else:
        PrecStr = "Precision is undefined"

    # Recall = TP/(FN+TP)
    if FNTotal + TPTotal > 0 and TPTotal >= 0:
        Rec = round(100 * accuracy_metrics["recall"], 1)
        RecStr = "Recall = " + str(Rec) + "%"
    else:
        RecStr = "Recall is undefined"

    # Total pixels
    Total = TPTotal + FPTotal + TNTotal + FNTotal

    PrintString = (
        "True Positive: "
        + str(f"{100 * TPTotal / Total:.2g}")
        + "%\nTrue Negative: "
        + str(f"{100 * TNTotal / Total:.2g}")
        + "%\nFalse Positive: "
        + str(f"{100 * FPTotal / Total:.2g}")
        + "%\nFalse Negative: "
        + str(f"{100 * FNTotal / Total:.2g}")
        + "%"
        + PrecStr
        + RecStr
    )

    TPStr = "True Positive: " + str(f"{100 * TPTotal / Total:.2g}") + "%"
    FNStr = "False Negative: " + str(f"{100 * FNTotal / Total:.2g}") + "%"
    FPStr = "False Positive: " + str(f"{100 * FPTotal / Total:.2g}") + "%"
    TNStr = "True Negative: " + str(f"{100 * TNTotal / Total:.2g}") + "%"

    uniqVals = np.unique(CombineArray)

    # Remove nan from unique values and convert remaining floats to ints
    colnums = uniqVals[~np.isnan(uniqVals)].astype(int)

    # Select only colours that correspond to data values in array
    colpac2 = [colpac[i] for i in colnums]

    #
    # Make final plot
    #
    fig, axes = plt.subplots(1, 1, figsize=(12, 12))
    plt.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.1)

    #CombineArray[0].plot(
    #    ax=axes, levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors=colpac2, add_colorbar=False
    #)

    # Example of using contourf instead of plot
    #contour = axes.contourf(
    #    CombineArray[0], levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors=colpac2
    #)

    # Add colorbar
    #cbar = fig.colorbar(contour, ax=axes)

    #plt.show()

    strDic = {0: TPStr, 1: FNStr, 2: FPStr, 3: TNStr, 4: PrecStr, 5: RecStr}
    # colDic = {0: colpac2[0], 1: colpac2[3], 2: colpac2[2], 3: colpac2[1]}

    for i in range(6):
        axes.text(
            1.04,
            0.98 - (0.04 * i),
            strDic[i],
            ha="left",
            va="center",
            size=14,
            color="k",
            transform=axes.transAxes,
        )
        if i < 4:
            rect = Rectangle(
                (1.01, 0.97 - (0.04 * i)),
                0.025,
                0.025,
                linewidth=1.5,
                edgecolor="black",
                facecolor=colpac[i + 1],
                clip_on=False,
                transform=axes.transAxes,
            )
            axes.add_patch(rect)

    axes.set_aspect("equal", adjustable="box")

    axes.set_title(FigTitle)

    # Define the filename of the output png image and export to png
    png_filepath = f"{save_folder}/{algo_name}_{study_site}-{year}-{year_basis}.png"
    plt.savefig(png_filepath, dpi=300)

    # Save combined array as tiff
    tif_filepath = f"{save_folder}/{algo_name}_{study_site}-{year}-{year_basis}.tif"

    print(CombineArray)

    write_cog(geo_im=CombineArray, fname=tif_filepath, overwrite=True)

    #
    # Write out metadata
    #
    metadata = {
        "Location": study_site.replace("_", " "),
        "Algorithm": algo_name,
        "Precision": Prec,
        "Recall": Rec,
        "Total Pixels": Total,
        "True Positive Pixels": TPTotal,
        "False Positive Pixels": FPTotal,
        "True Negative Pixels": TNTotal,
        "False Negative Pixels": FNTotal,
    }

    metadata_dict = {
        "Location": [study_site.replace("_", " ")],
        "Algorithm": [algo_name],
        "Precision": [Prec],
        "Recall": [Rec],
        "Total Pixels": [Total],
        "True Positive Pixels": [TPTotal],
        "False Positive Pixels": [FPTotal],
        "True Negative Pixels": [TNTotal],
        "False Negative Pixels": [FNTotal],
    }

    import pandas as pd

    result_df = pd.DataFrame.from_dict(metadata_dict)

    with open(
        f"{save_folder}/{algo_name}_{study_site}-{year}-{year_basis}.txt", "w"
    ) as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

    result_df.to_csv(
        f"{save_folder}/{algo_name}_{study_site}-{year}-{year_basis}.csv", index=False
    )

    climate_df = validation_climate_analysis(
        tif_filepath, ClimateZoneShapeFile, ClimateLegend_fname, study_site, vali_year
    )

    # Visualise the results in table format
    climate_df.style.pipe(result_visuals, study_site, vali_year)

    # Get the climate classes and their corresponding precision and recall values
    classes = climate_df.Name
    precisions = climate_df["Precision"].values
    recalls = climate_df["Recall"].values
    areas = climate_df["Area_Percent"].values

    # Set the width of the bars
    barWidth = 0.25

    # Set the position of the bars on the x-axis
    r1 = np.arange(len(classes))
    r2 = [x + barWidth for x in r1]
    r3 = [x + 2 * barWidth for x in r1]

    # Create the bar plot
    plt.bar(
        r1,
        precisions,
        color="orange",
        width=barWidth,
        edgecolor="white",
        label="Precision",
    )
    plt.bar(
        r2, recalls, color="purple", width=barWidth, edgecolor="white", label="Recall"
    )
    plt.bar(
        r3,
        areas,
        color="green",
        width=barWidth,
        edgecolor="white",
        label="Relative Area",
    )

    # Set the y-axis to show percentages (0-100%)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))

    # Add xticks on the middle of the group bars
    plt.xlabel("Climate Class", fontweight="bold")
    plt.ylabel("%", fontweight="bold")
    plt.xticks([r + barWidth for r in range(len(classes))], classes)

    # Add legend
    plt.legend()

    # Add title
    plt.title(
        "Precision, Recall, and Relative Area of the\n Validated Burnt Area Analysis by Climate Zone\n{}, {}, {}".format(
            study_site.replace("_", " "), vali_year, algo_name
        )
    )

    # Save the plot as a PNG image next to GeoTIFF files
    save_folder = "/".join(tif_filepath.split("/")[:-1])
    # save_name = tif_filepath.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    plt.savefig(
        f"{save_folder}/{algo_name}_{study_site}-{year}-{year_basis}_climate_analysis_graph.png"
    )

    # Display the plot
    # plt.show()

    # Export the results from results_df to csv
    # Define csv fname
    csv_fname = f"{save_folder}/{algo_name}_{study_site}-{year}-{year_basis}_Climate_Validation_statistics.csv"
    # Export df
    climate_df.to_csv(csv_fname, index=False)


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
    "--method-result-folder",
    "-m",
    type=str,
    default=None,
    help="REQUIRED. The S3 URL which refers to folder which includes output files.",
)
@click.option(
    "--performance-report-output-folder",
    "-o",
    type=str,
    default=None,
    help="REQUIRED. The output S3 URL to save the performance reports.",
)
@click.option(
    "--suffix",
    "-s",
    type=str,
    default="demo",
    help="The suffix of method result GeoTIFF.",
)
@click.option(
    "--year-basis",
    "-y",
    type=str,
    default=None,
    help="The CY or FY.",
)
def run_validation(
    method_result_folder,
    performance_report_output_folder,
    suffix,
    year_basis,
):

    # download all ground truth files from AWS S3
    cfg_folder = "https://dea-public-data-dev.s3.ap-southeast-2.amazonaws.com/projects/burn_cube/configs/"

    file_names = [
        "Port_Hedland_WA_2020_CY_ARDC_polygons.gpkg",
        "Esperance_WA_2020_CY_ARDC_polygons.gpkg",
        "East_Vic_2020_CY_ARDC_polygons.gpkg",
        "Cairns_QLD_2020_CY_ARDC_polygons.gpkg",
        "kangaroo_Island_SA_2020_CY_ARDC_polygons.gpkg",
        "Cooktown_QLD_2020_CY_ARDC_polygons.gpkg",
        "ITEMCoastlineCleaned.gpkg",
        "Koppen_Climate_Zones.gpkg",
    ]

    for file_name in file_names:

        # URL of the public S3 object
        s3_url = cfg_folder + file_name

        download_file_from_s3_public(s3_url, file_name)

    # Set AWS environment variable
    os.environ["AWS_NO_SIGN_REQUEST"] = "yes"

    # Define required user inputs
    # Examples will be given using the following s3 object path
    # http://dea-public-data-dev.s3-website-ap-southeast-2.amazonaws.com/?prefix=projects/burn_cube/derivative/ga_ls8c_nbart_bc_cyear_3/
    # 3-0-0/x14/y21/2020-01-01--P1Y/ga_ls8c_nbart_bc_cyear_3_x14y21_2020-01-01--P1Y_final_wofssevere.tif

    # Define the files type
    # E.g. 'tif'
    inputType = "tif"

    # What is the path to the coastline dataset?
    CoastLineShapeFile = "ITEMCoastlineCleaned.gpkg"

    # Define the colour scheme to be used in validation figures
    colpac = ["#000000", "#e69f00", "#57b4e9", "#019e73", "#f0e442"]

    # What year are we analysing? If using a FY basis, enter the first year over the period (i.e. for 19/20 enter 2019)
    year = 2020

    # If calculating on a CY year basis, do you want to extend the validation period to capture fires
    # that ignited in the early summer months (i.e. 1 for december and 2 for novemeber)?
    extra_months = 1

    # Define a location name and year for your validation AOI. Use '_' instead of spaces.
    vali_year = str(year)

    # Set filepath for the Climate Zone Shapefile and legend
    ClimateZoneShapeFile = "Koppen_Climate_Zones.gpkg"
    ClimateLegend_fname = "ardc_historic_burn/Burnt_Area_Validation/legend.txt"

    Kangaroo_Island_grid_codes = ["x32y15", "x32y16", "x33y15", "x33y16"]
    East_VIC_grid_codes = [
        "x41y13",
        "x41y14",
        "x42y13",
        "x42y14",
        "x43y13",
        "x43y14",
        "x44y13",
        "x44y14",
    ]
    FNQ_grid_codes = ["x40y39", "x41y39", "x42y39"]
    Port_Hedland_grid_codes = ["x14y32", "x14y33", "x15y32", "x15y33"]
    Esperance_codes = [
        "x15y18",
        "x15y19",
        "x15y20",
        "x16y18",
        "x16y19",
        "x16y20",
        "x17y18",
        "x17y19",
        "x17y20",
        "x18y18",
        "x18y19",
        "x18y20",
        "x19y18",
        "x19y19",
        "x19y20",
        "x20y18",
        "x20y19",
        "x20y20",
    ]
    Cairns_grid_codes = ["x42y35", "x42y36", "x42y37", "x43y35", "x43y36", "x43y37"]

    # study_sites = ["kangaroo_Island_SA", "East_Vic", "Cairns_QLD", "Cooktown_QLD", "Esperance_WA", "Port_Hedland_WA"]

    all_study_sites_codes = {
        "kangaroo_Island_SA": Kangaroo_Island_grid_codes,
        "East_Vic": East_VIC_grid_codes,
        "Cairns_QLD": Cairns_grid_codes,
        "Cooktown_QLD": FNQ_grid_codes,
        "Esperance_WA": Esperance_codes,
        "Port_Hedland_WA": Port_Hedland_grid_codes,
    }

    # Download GeoTIFF files from root folder, and save them to study stie folders
    # e.g., s3://dea-public-data-dev/projects/burn_cube/derivative/ga_ls8c_nbart_bc_7bands_4years_cyear_3

    from pathlib import Path

    import s3fs

    # Initialize an S3 filesystem object
    fs = s3fs.S3FileSystem()

    # method_result_folder = "s3://dea-public-data-dev/projects/burn_cube/derivative/ga_ls8c_nbart_vic_rf_cyear_3"

    s3_path = method_result_folder + f"/**_{suffix}.tif"

    # Create the full S3 path to list files from

    algo_name = s3_path.split("derivative")[-1].split("/")[1]

    # List all files under the specified S3 prefix
    files = fs.glob(s3_path)

    for study_site_name, study_site_codes in all_study_sites_codes.items():

        study_region_files = []

        for code in study_site_codes:
            study_region_files.extend([file for file in files if code in file])

        local_folder = f"performance-report/{study_site_name}_{algo_name}"

        print(local_folder)

        path = Path(local_folder)
        path.mkdir(parents=True, exist_ok=True)

        for study_region_file in study_region_files:

            local_file_path = f"{local_folder}/{study_region_file.split('/')[-1]}"
            print(study_region_file, "====>", local_file_path)

            #  Open the S3 file and read its contents, then write it to a local file
            with fs.open(study_region_file, "rb") as s3_file:
                with open(local_file_path, "wb") as local_file:
                    local_file.write(s3_file.read())

    study_sites = [
        "Esperance_WA",
        "Port_Hedland_WA",
        "kangaroo_Island_SA",
        "East_Vic",
        "Cairns_QLD",
        "Cooktown_QLD",
    ]

    for study_site in study_sites:
        save_folder = f"performance-report/{study_site}_{algo_name}"

        generate_result_by_study_site_folder(
            algo_name,
            save_folder,
            study_site,
            year_basis,
            extra_months,
            CoastLineShapeFile,
            suffix,
            inputType,
            colpac,
            ClimateZoneShapeFile,
            ClimateLegend_fname,
            vali_year,
        )

    import pandas as pd

    overall_result = []
    overall_climate_list = []

    for study_site in study_sites:
        save_folder = f"performance-report/{study_site}_{algo_name}"
        csv_file_name = f"{algo_name}_{study_site}-{year}-{year_basis}.csv"
        climate_file_name = f"{algo_name}_{study_site}-{year}-{year_basis}_Climate_Validation_statistics.csv"

        # result CSV
        overall_result.append(pd.read_csv(f"{save_folder}/{csv_file_name}"))

        climate_df = pd.read_csv(f"{save_folder}/{climate_file_name}")
        climate_df["study site"] = [study_site] * len(climate_df)
        overall_climate_list.append(climate_df)

    overall_df = pd.concat(overall_result)

    overall_df.to_csv(f"performance-report/{algo_name}-overall.csv", index=False)

    overall_climate_df = pd.concat(overall_climate_list)

    overall_climate_df.to_csv(
        f"performance-report/{algo_name}-climate-class-overall.csv", index=False
    )

    bucket_name, s3_folder = performance_report_output_folder.replace(
        "s3://", ""
    ).split("/", 1)

    print("s3_folder", s3_folder)

    helper.get_and_set_aws_credentials()

    upload_folder_to_s3("performance-report", bucket_name, s3_folder)


if __name__ == "__main__":
    run_validation()
