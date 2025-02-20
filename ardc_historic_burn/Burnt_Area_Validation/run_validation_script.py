#!/usr/bin/env python3
"""
Improved AWS Download and ARDC Dataset Extraction/Validation Script

This script downloads required ground truth files, processes GeoTIFFs,
calculates validation metrics and climate statistics, generates plots and tables,
and finally uploads performance reports to S3.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import click
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import xarray as xr
from datacube.utils.cog import write_cog
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
import s3fs

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
from dea_burn_cube import helper


def upload_folder_to_s3(local_folder_path: str, bucket_name: str, s3_folder: str) -> None:
    """Uploads a local folder to an S3 bucket.

    Args:
        local_folder_path (str): Local folder path.
        bucket_name (str): S3 bucket name.
        s3_folder (str): Destination folder in the S3 bucket.
    """
    s3_client = boto3.client("s3")
    for subdir, _, files in os.walk(local_folder_path):
        for file in files:
            full_path = os.path.join(subdir, file)
            s3_path = os.path.join(s3_folder, os.path.relpath(full_path, local_folder_path))
            print(f"Uploading {full_path} to s3://{bucket_name}/{s3_path}")
            s3_client.upload_file(full_path, bucket_name, s3_path)


def result_visuals(styler: Any, study_site: str, vali_year: str) -> Any:
    """Styles a Pandas DataFrame for visualizing climate validation results.

    Args:
        styler: A Pandas Styler object.
        study_site (str): Study site name.
        vali_year (str): Validation year.

    Returns:
        The styled DataFrame.
    """
    caption = f"Climate Class Validation Results for {study_site.replace('_', ' ')}, {vali_year}"
    styler.set_caption(caption)
    styler.format({
        "Precision": lambda x: f"{x * 100:.1f}%",
        "Recall": lambda x: f"{x * 100:.1f}%",
        "Area_Percent": lambda x: f"{x * 100:.1f}%"
    })
    styler.background_gradient(subset=["Precision", "Recall"], vmin=0.3, vmax=1, cmap="RdYlGn")
    return styler

def generate_result_by_study_site_folder(
    algo_name: str,
    save_folder: str,
    study_site: str,
    year_basis: str,
    extra_months: int,
    coastline_shp: str,
    suffix: str,
    input_type: str,
    colpac: List[str],
    climate_zone_shp: str,
    climate_legend_fname: str,
    vali_year: str,
) -> None:
    """Generate results and plots for a study site folder.

    Args:
        algo_name (str): Algorithm name.
        save_folder (str): Local folder where TIFF files are stored.
        study_site (str): Study site name.
        year_basis (str): "CY" or "FY".
        extra_months (int): Extra months for calendar year.
        coastline_shp (str): Path to coastline shapefile.
        suffix (str): TIFF suffix filter.
        input_type (str): Input file type ('tif', etc.).
        colpac (List[str]): List of color codes.
        climate_zone_shp (str): Path to climate zone shapefile.
        climate_legend_fname (str): Path to climate legend file.
        vali_year (str): Validation year string.
    """
    year = 2020

    # Get bounding polygon and CRS from TIFF files
    polygon, poly_crs = raster_folder_bbox(save_folder)

    _ = ardc_year_calc(year_basis, year, extra_months)  # time period printed by function

    # Create GeoDataFrame from polygon and reproject to EPSG:3577
    poly_gdf = gpd.GeoDataFrame(geometry=[polygon], crs=poly_crs).to_crs(epsg=3577)

    geopackage_outpath = f"{study_site}_{year}_{year_basis}_ARDC_polygons.gpkg"
    _ = gpd.read_file(geopackage_outpath)  # read if needed later
    fig_title = f"{algo_name} Results for {study_site} {year}-{year_basis}"
    ground_truth_file = geopackage_outpath
    gtf, clsf, state_bndry = read_shapes(ground_truth_file, coastline_shp)

    tif_list = get_tifs(save_folder, suffix)
    tif_coords = [extract_xy(tif) for tif in tif_list if extract_xy(tif)]
    print(study_site, algo_name, tif_list)

    # Process each TIFF to merge validation statistics
    combine_list = []
    tp_total = fn_total = fp_total = tn_total = 0
    for tif_path in tif_list:
        comby, tp, fn, fp, tn = validation_stats(
            tif_path, gtf, clsf, state_bndry, input_type, colpac, graph_out=False
        )
        comby.name = "var"
        combine_list.append(comby)
        tp_total += tp
        fn_total += fn
        fp_total += fp
        tn_total += tn

    try:
        combine_array = xr.merge(combine_list).to_array()
        print("combine_array created by xarray merge")
    except xr.MergeError:
        combine_array = xr.combine_by_coords(combine_list).to_array()
        print("combine_array created by xarray combine_by_coords")

    # Extend the metrics list to include accuracy, balanced_accuracy, and fpr
    metric_list = ["accuracy", "balanced_accuracy", "fpr", "precision", "recall", "f1-score"]
    accuracy_metrics = calculate_classification_metrics(
        tp=tp_total, tn=tn_total, fp=fp_total, fn=fn_total, metrics=metric_list
    )
    print(study_site, algo_name, accuracy_metrics)

    total_pixels = tp_total + fp_total + tn_total + fn_total

    # Calculate and format each metric
    if tp_total + fp_total > 0 and tp_total >= 0:
        precision_val = round(100 * accuracy_metrics["precision"], 1)
        precision_str = f"Precision = {precision_val}%"
    else:
        precision_val = None
        precision_str = "Precision is undefined"

    if fn_total + tp_total > 0 and tp_total >= 0:
        recall_val = round(100 * accuracy_metrics["recall"], 1)
        recall_str = f"Recall = {recall_val}%"
    else:
        recall_val = None
        recall_str = "Recall is undefined"

    # New metrics added:
    if total_pixels > 0:
        accuracy_val = round(100 * accuracy_metrics["accuracy"], 1)
        accuracy_str = f"Accuracy = {accuracy_val}%"
        balanced_accuracy_val = round(100 * accuracy_metrics["balanced_accuracy"], 1)
        balanced_accuracy_str = f"Balanced Accuracy = {balanced_accuracy_val}%"
    else:
        accuracy_val = None
        accuracy_str = "Accuracy is undefined"
        balanced_accuracy_val = None
        balanced_accuracy_str = "Balanced Accuracy is undefined"

    if (tn_total + fp_total) > 0:
        fpr_val = round(100 * accuracy_metrics["fpr"], 1)
        fpr_str = f"FPR = {fpr_val}%"
    else:
        fpr_val = None
        fpr_str = "FPR is undefined"

    tp_str = f"True Positive: {100 * tp_total / total_pixels:.2g}%"
    fn_str = f"False Negative: {100 * fn_total / total_pixels:.2g}%"
    fp_str = f"False Positive: {100 * fp_total / total_pixels:.2g}%"
    tn_str = f"True Negative: {100 * tn_total / total_pixels:.2g}%"

    # Choose colors based on unique values in combined array
    uniq_vals = np.unique(combine_array)
    colnums = uniq_vals[~np.isnan(uniq_vals)].astype(int)
    colpac2 = [colpac[i] for i in colnums if i < len(colpac)]

    # Plot final combined result with metrics annotations
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.1)

    # Extend metrics_dict to include the three new metrics (keys 6, 7, 8)
    metrics_dict = {
        0: tp_str,
        1: fn_str,
        2: fp_str,
        3: tn_str,
        4: precision_str,
        5: recall_str,
        6: accuracy_str,
        7: balanced_accuracy_str,
        8: fpr_str,
    }
    for i in range(9):
        ax.text(1.04, 0.98 - (0.04 * i), metrics_dict[i],
                ha="left", va="center", fontsize=14, color="k", transform=ax.transAxes)
        if i < 4:
            rect = Rectangle((1.01, 0.97 - (0.04 * i)), 0.025, 0.025,
                             linewidth=1.5, edgecolor="black", facecolor=colpac[i + 1],
                             clip_on=False, transform=ax.transAxes)
            ax.add_patch(rect)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(fig_title)

    png_filepath = Path(save_folder) / f"{algo_name}_{study_site}-{year}-{year_basis}.png"
    plt.savefig(png_filepath, dpi=300)

    tif_filepath = Path(save_folder) / f"{algo_name}_{study_site}-{year}-{year_basis}.tif"
    print(combine_array)
    write_cog(geo_im=combine_array, fname=str(tif_filepath), overwrite=True)

    # Write metadata including the new metrics
    metadata: Dict[str, Any] = {
        "Location": study_site.replace("_", " "),
        "Algorithm": algo_name,
        "Accuracy": accuracy_val,
        "Balanced Accuracy": balanced_accuracy_val,
        "FPR": fpr_val,
        "Precision": precision_val,
        "Recall": recall_val,
        "Total Pixels": total_pixels,
        "True Positive Pixels": tp_total,
        "False Positive Pixels": fp_total,
        "True Negative Pixels": tn_total,
        "False Negative Pixels": fn_total,
    }
    meta_txt = Path(save_folder) / f"{algo_name}_{study_site}-{year}-{year_basis}.txt"
    with open(meta_txt, "w") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

    result_df = pd.DataFrame.from_dict({
        "Location": [study_site.replace("_", " ")],
        "Algorithm": [algo_name],
        "Accuracy": [accuracy_val],
        "Balanced Accuracy": [balanced_accuracy_val],
        "FPR": [fpr_val],
        "Precision": [precision_val],
        "Recall": [recall_val],
        "Total Pixels": [total_pixels],
        "True Positive Pixels": [tp_total],
        "False Positive Pixels": [fp_total],
        "True Negative Pixels": [tn_total],
        "False Negative Pixels": [fn_total],
    })
    result_csv = Path(save_folder) / f"{algo_name}_{study_site}-{year}-{year_basis}.csv"
    result_df.to_csv(result_csv, index=False)

    # Climate analysis and visualisation
    climate_df = validation_climate_analysis(
        str(tif_filepath), climate_zone_shp, climate_legend_fname, study_site, vali_year
    )
    climate_df.style.pipe(result_visuals, study_site, vali_year)

    # Bar plot of climate metrics
    classes = climate_df["Name"]
    precisions = climate_df["Precision"].values
    recalls = climate_df["Recall"].values
    areas = climate_df["Area_Percent"].values

    bar_width = 0.25
    r1 = np.arange(len(classes))
    r2 = r1 + bar_width
    r3 = r1 + 2 * bar_width

    plt.figure()
    plt.bar(r1, precisions, color="orange", width=bar_width, edgecolor="white", label="Precision")
    plt.bar(r2, recalls, color="purple", width=bar_width, edgecolor="white", label="Recall")
    plt.bar(r3, areas, color="green", width=bar_width, edgecolor="white", label="Relative Area")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.xlabel("Climate Class", fontweight="bold")
    plt.ylabel("%", fontweight="bold")
    plt.xticks(r1 + bar_width, classes)
    plt.legend()
    plt.title(f"Precision, Recall, and Relative Area of the\n Validated Burnt Area Analysis by Climate Zone\n{study_site.replace('_', ' ')}, {vali_year}, {algo_name}")
    
    climate_png = Path(save_folder) / f"{algo_name}_{study_site}-{year}-{year_basis}_climate_analysis_graph.png"
    plt.savefig(climate_png)
    
    csv_fname = Path(save_folder) / f"{algo_name}_{study_site}-{year}-{year_basis}_Climate_Validation_statistics.csv"
    climate_df.to_csv(csv_fname, index=False)


def download_file_from_s3_public(url: str, file_path: str) -> None:
    """Download a file from a public S3 URL.

    Args:
        url (str): Public URL.
        file_path (str): Local file path to save the content.
    """
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"File downloaded successfully from: {url}")
    else:
        print(f"Failed to download file from: {url}")


@click.command(no_args_is_help=True)
@click.option("--method-result-folder", "-m", type=str, required=True,
              help="REQUIRED. The S3 URL referring to the folder with output files.")
@click.option("--performance-report-output-folder", "-o", type=str, required=True,
              help="REQUIRED. The output S3 URL to save the performance reports.")
@click.option("--suffix", "-s", type=str, default="demo",
              help="The suffix of method result GeoTIFF.")
@click.option("--year-basis", "-y", type=str, required=True,
              help="The CY or FY.")
def run_validation(method_result_folder: str, performance_report_output_folder: str,
                   suffix: str, year_basis: str) -> None:
    """Main function to run validation processing.

    It downloads required ground truth files, processes study sites, generates reports,
    and uploads performance reports to S3.
    """
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
        s3_url = f"{cfg_folder}{file_name}"
        download_file_from_s3_public(s3_url, file_name)

    os.environ["AWS_NO_SIGN_REQUEST"] = "yes"

    # User-defined inputs
    input_type = "tif"
    coastline_shp = "ITEMCoastlineCleaned.gpkg"
    colpac = ["#000000", "#e69f00", "#57b4e9", "#019e73", "#f0e442"]
    year = 2020
    extra_months = 1
    vali_year_str = str(year)
    climate_zone_shp = "Koppen_Climate_Zones.gpkg"
    climate_legend_fname = "ardc_historic_burn/Burnt_Area_Validation/legend.txt"

    kangaroo_island_codes = ["x32y15", "x32y16", "x33y15", "x33y16"]
    east_vic_codes = ["x41y13", "x41y14", "x42y13", "x42y14", "x43y13", "x43y14", "x44y13", "x44y14"]
    fnq_codes = ["x40y39", "x41y39", "x42y39"]
    port_hedland_codes = ["x14y32", "x14y33", "x15y32", "x15y33"]
    esperance_codes = ["x15y18", "x15y19", "x15y20", "x16y18", "x16y19", "x16y20",
                       "x17y18", "x17y19", "x17y20", "x18y18", "x18y19", "x18y20",
                       "x19y18", "x19y19", "x19y20", "x20y18", "x20y19", "x20y20"]
    cairns_codes = ["x42y35", "x42y36", "x42y37", "x43y35", "x43y36", "x43y37"]

    all_study_sites_codes: Dict[str, List[str]] = {
        "kangaroo_Island_SA": kangaroo_island_codes,
        "East_Vic": east_vic_codes,
        "Cairns_QLD": cairns_codes,
        "Cooktown_QLD": fnq_codes,
        "Esperance_WA": esperance_codes,
        "Port_Hedland_WA": port_hedland_codes,
    }

    fs = s3fs.S3FileSystem()
    s3_path = f"{method_result_folder}/**_{suffix}.tif"
    algo_name = s3_path.split("derivative")[-1].split("/")[1]
    files = fs.glob(s3_path)

    # Download study site files
    for site_name, site_codes in all_study_sites_codes.items():
        study_region_files = [file for file in files if any(code in file for code in site_codes)]
        local_folder = Path(f"performance-report/{site_name}_{algo_name}")
        local_folder.mkdir(parents=True, exist_ok=True)
        for s3_file_path in study_region_files:
            local_file_path = local_folder / Path(s3_file_path).name
            print(f"{s3_file_path} ==> {local_file_path}")
            with fs.open(s3_file_path, "rb") as s3_file, open(local_file_path, "wb") as local_file:
                local_file.write(s3_file.read())

    study_sites = ["Esperance_WA", "Port_Hedland_WA", "kangaroo_Island_SA", "East_Vic", "Cairns_QLD", "Cooktown_QlD"]
    for site in study_sites:
        site_folder = f"performance-report/{site}_{algo_name}"
        generate_result_by_study_site_folder(
            algo_name=algo_name,
            save_folder=site_folder,
            study_site=site,
            year_basis=year_basis,
            extra_months=extra_months,
            coastline_shp=coastline_shp,
            suffix=suffix,
            input_type=input_type,
            colpac=colpac,
            climate_zone_shp=climate_zone_shp,
            climate_legend_fname=climate_legend_fname,
            vali_year=vali_year_str,
        )

    overall_results = []
    overall_climate_list = []
    for site in study_sites:
        site_folder = Path(f"performance-report/{site}_{algo_name}")
        result_csv = site_folder / f"{algo_name}_{site}-{year}-{year_basis}.csv"
        climate_csv = site_folder / f"{algo_name}_{site}-{year}-{year_basis}_Climate_Validation_statistics.csv"
        overall_results.append(pd.read_csv(result_csv))
        climate_df = pd.read_csv(climate_csv)
        climate_df["study site"] = site
        overall_climate_list.append(climate_df)

    overall_df = pd.concat(overall_results, ignore_index=True)
    overall_df.to_csv(f"performance-report/{algo_name}-{suffix}-overall.csv", index=False)

    overall_climate_df = pd.concat(overall_climate_list, ignore_index=True)
    overall_climate_df.to_csv(f"performance-report/{algo_name}-climate-class-{suffix}-overall.csv", index=False)

    bucket_name, s3_folder = performance_report_output_folder.replace("s3://", "").split("/", 1)
    print("s3_folder", s3_folder)
    helper.get_and_set_aws_credentials()
    upload_folder_to_s3("performance-report", bucket_name, s3_folder)


if __name__ == "__main__":
    run_validation()
