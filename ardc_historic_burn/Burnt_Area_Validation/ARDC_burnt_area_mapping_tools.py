#!/usr/bin/env python3
"""
Improved ARDC Burnt Area Mapping Tools

This module provides tools for processing burnt area mapping,
including grid code generation, climate data import, S3 file download,
validation statistics, and climate analysis.

Improvements:
- Better code styling following PEP8 and pylint guidelines.
- Enhanced performance with list comprehensions and efficient operations.
- Added type hints for clarity.
"""

import os
import re
import time
from datetime import datetime
from typing import List, Tuple, Optional, Union, Dict, Any

import boto3
import botocore
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rioxarray
import xarray as xr
from datacube.testutils.io import rio_slurp_xarray
from dea_tools.spatial import xr_rasterize
from rasterio import features
from shapely.geometry import Polygon, shape

# Set environment variable for AWS S3 access without credentials
os.environ["AWS_NO_SIGN_REQUEST"] = "yes"


def gen_grid_codes(x_range: Tuple[int, int], y_range: Tuple[int, int]) -> List[str]:
    """
    Generate a list of grid codes in the format 'x{num}y{num}' for given x and y ranges.

    Args:
        x_range (Tuple[int, int]): (min_x, max_x) range.
        y_range (Tuple[int, int]): (min_y, max_y) range.

    Returns:
        List[str]: List of grid codes.
    """
    return [f"x{i}y{j}" for i in range(x_range[0], x_range[1] + 1)
            for j in range(y_range[0], y_range[1] + 1)]


def koppen_import(koppen_fname: str, legend_fname: str) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Import Koppen-Geiger climate zones and legend.

    Args:
        koppen_fname (str): Path to the Koppen-Geiger GeoPackage.
        legend_fname (str): Path to the Koppen-Geiger legend text file.

    Returns:
        Tuple[gpd.GeoDataFrame, pd.DataFrame]: Merged GeoDataFrame and legend DataFrame.
    """
    # Read legend file
    with open(legend_fname, 'r') as f:
        lines = f.readlines()

    pattern = re.compile(r"(\d+):\s+(\w+)\s+(.*)\s+\[(\d+)\s+(\d+)\s+(\d+)\]")
    codes, names, descriptions, colors = [], [], [], []

    for line in lines[3:]:  # Skip header lines
        match = pattern.search(line)
        if match:
            codes.append(int(match.group(1)))
            names.append(match.group(2))
            descriptions.append(match.group(3))
            colors.append(tuple(int(v) / 255.0 for v in (match.group(4), match.group(5), match.group(6))))

    legend_df = pd.DataFrame({
        "gridcode": codes,
        "Name": names,
        "Description": descriptions,
        "Color": colors
    })

    climate_gdf = gpd.read_file(koppen_fname)
    climate_gdf = pd.merge(climate_gdf, legend_df, on="gridcode")
    return climate_gdf, legend_df


def download_s3_files(bucket_name: str, path_to_download: str, save_as: Optional[str] = None) -> None:
    """
    Download a file from an Amazon S3 bucket.

    Args:
        bucket_name (str): S3 bucket name.
        path_to_download (str): File path in the S3 bucket.
        save_as (Optional[str]): Local filename to save as. Defaults to original name.
    """
    client = boto3.client("s3", config=botocore.client.Config(signature_version=botocore.UNSIGNED))
    target = save_as if save_as else path_to_download
    client.download_file(bucket_name, path_to_download, target)


def ardc_year_calc(year_basis: str, year: int, extra_months: int = 0) -> Tuple[str, str]:
    """
    Calculate start and end dates for a validation period based on year basis.

    Args:
        year_basis (str): "FY" for Fiscal Year or "CY" for Calendar Year.
        year (int): Year value.
        extra_months (int, optional): Extra months for CY basis. Defaults to 0.

    Returns:
        Tuple[str, str]: (start_date, end_date) formatted as "%Y-%m-%d".

    Raises:
        ValueError: If year_basis is not "FY" or "CY".
    """
    if year_basis not in ("FY", "CY"):
        raise ValueError("Invalid year basis. Must be 'FY' or 'CY'.")

    if year_basis == "FY":
        start_date = datetime(year=year, month=7, day=1)
        end_date = datetime(year=year + 1, month=6, day=30)
    else:
        start_date = datetime(year=year, month=1, day=1) if extra_months == 0 else datetime(year=year - 1, month=13 - extra_months, day=1)
        end_date = datetime(year=year, month=12, day=31)

    formatted_start = start_date.strftime("%Y-%m-%d")
    formatted_end = end_date.strftime("%Y-%m-%d")
    print(f"Start Date: {formatted_start}")
    print(f"End Date: {formatted_end}")
    return formatted_start, formatted_end


def raster_folder_bbox(save_folder: str) -> Tuple[Polygon, Any]:
    """
    Compute bounding box polygon and CRS from all TIFF files in a folder.

    Args:
        save_folder (str): Directory path containing TIFF files.

    Returns:
        Tuple[Polygon, Any]: Bounding box polygon and CRS.
    """
    min_x, min_y = float("inf"), float("inf")
    max_x, max_y = float("-inf"), float("-inf")
    poly_crs = None

    for filename in os.listdir(save_folder):
        if filename.lower().endswith(".tif"):
            file_path = os.path.join(save_folder, filename)
            with rasterio.open(file_path) as src:
                bounds = src.bounds
                min_x = min(min_x, bounds.left)
                min_y = min(min_y, bounds.bottom)
                max_x = max(max_x, bounds.right)
                max_y = max(max_y, bounds.top)
                poly_crs = src.crs

    bbox = Polygon([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])
    return bbox, poly_crs


def read_shapes(
    ground_truth_file: str,
    coast_line_file: str,
    state_bndry_file: Optional[str] = None,
    state: str = "",
    column_filter: Optional[str] = None,
    filter_entries: Optional[List] = None
) -> Tuple[Union[gpd.GeoDataFrame, xr.DataArray], gpd.GeoDataFrame, Union[gpd.GeoDataFrame, str]]:
    """
    Read shapefiles and optionally subset based on given criteria.

    Args:
        ground_truth_file (str): Path to the ground truth file (shapefile or GeoTIFF).
        coast_line_file (str): Path to the coast line shapefile.
        state_bndry_file (Optional[str], optional): Path to the state boundary file. Defaults to None.
        state (str, optional): State name for subsetting. Defaults to "".
        column_filter (Optional[str], optional): Column name for filtering. Defaults to None.
        filter_entries (Optional[List], optional): List of filter entries. Defaults to None.

    Returns:
        Tuple: (subset ground truth, coastline shapefile, state boundary or empty string)
    """
    _, ext = os.path.splitext(ground_truth_file.lower())
    if ext == ".gpkg":
        gtf = gpd.read_file(ground_truth_file)
    elif ext == ".tif":
        gtf = rio_slurp_xarray(ground_truth_file)
        if getattr(gtf, "spatial_ref", None) == 3111:
            gtf = gtf.rio.reproject("EPSG:3577")
    else:
        raise ValueError("Ground Truth File must be a geopackage or GeoTIFF.")

    if column_filter and filter_entries is not None and column_filter in gtf.columns:
        gtf_sub = gtf[gtf[column_filter].isin(filter_entries)]
    else:
        gtf_sub = gtf

    clsf = gpd.read_file(coast_line_file)

    if state and state_bndry_file:
        bndry = gpd.read_file(state_bndry_file)
        state_bndry = bndry[bndry.STE_NAME21 == state].to_crs("EPSG:3577")
    else:
        state_bndry = ""
    return gtf_sub, clsf, state_bndry


def validation_stats(
    product: str,
    gtf_sub: Union[gpd.GeoDataFrame, xr.DataArray],
    clsf: gpd.GeoDataFrame,
    state_bndry: Union[gpd.GeoDataFrame, str],
    input_type: str,
    colpac: List,
    graph_out: bool = True
) -> Tuple[xr.DataArray, int, int, int, int]:
    """
    Compute validation statistics comparing a product raster with ground truth data.

    Args:
        product (str): Path to the product file.
        gtf_sub (Union[gpd.GeoDataFrame, xr.DataArray]): Ground truth subset.
        clsf (gpd.GeoDataFrame): Coastline shapefile.
        state_bndry (Union[gpd.GeoDataFrame, str]): State boundary data.
        input_type (str): Type of input ('tif' or other).
        colpac (List): Color palette list.
        graph_out (bool, optional): Whether to output graphs. Defaults to True.

    Returns:
        Tuple: (combined result, TP, FN, FP, TN)
    """
    if input_type == "tif":
        prod_array = rio_slurp_xarray(product)
    else:
        prod_array = rioxarray.open_rasterio(product).Moderate

    if isinstance(state_bndry, (gpd.GeoDataFrame, gpd.GeoSeries)) and not state_bndry.empty:
        state_mask = xr_rasterize(state_bndry, prod_array)
        oc_mask = xr_rasterize(clsf, prod_array)
        ocean_mask = np.logical_and(oc_mask, state_mask)
    else:
        ocean_mask = xr_rasterize(clsf, prod_array)

    masked_ocean = prod_array.where(ocean_mask)

    if isinstance(gtf_sub, (pd.DataFrame, gpd.GeoDataFrame)):
        gtf_sub_array = xr_rasterize(gtf_sub, prod_array)
        mask = gtf_sub_array
        temp_mask = masked_ocean.where(gtf_sub_array == 1)
    else:
        temp_mask = gtf_sub.rio.reproject_match(prod_array)
        mask = temp_mask.where(temp_mask == 0, 1)

    tp = int(masked_ocean.where((masked_ocean == 1) & (mask == 1)).count())
    fn = int(masked_ocean.where((masked_ocean == 0) & (mask == 1)).count())
    fp = int(masked_ocean.where((masked_ocean == 1) & (mask == 0)).count())
    tn = int(masked_ocean.where((masked_ocean == 0) & (mask == 0)).count())

    prec_str = (f"\nPrecision = {round(100 * tp / (tp + fp), 1)}%"
                if (tp + fp > 0 and tp > 0) else "\nPrecision is undefined")
    rec_str = (f"\nRecall = {round(100 * tp / (tp + fn), 1)}%"
               if (tp + fn > 0 and tp > 0) else "\nRecall is undefined")
    print_string = (f"True Positives = {tp}\nTrue Negatives = {tn}\n"
                    f"False Positives = {fp}\nFalse Negatives = {fn}\n"
                    f"{prec_str}{rec_str}")

    # Merge arrays for visualization
    tru_pos = masked_ocean.where((masked_ocean == 1) & (mask == 1))
    fal_neg = (masked_ocean.where((masked_ocean == 0) & (mask == 1)) + 1) * 2
    fal_pos = masked_ocean.where((masked_ocean == 1) & (mask == 0)) * 3
    tru_neg = (masked_ocean.where((masked_ocean == 0) & (mask == 0)) + 1) * 4
    tru_pos.name, fal_neg.name, fal_pos.name, tru_neg.name = "TP", "FN", "FP", "TN"
    merged = xr.merge([tru_pos.fillna(0), fal_neg.fillna(0), fal_pos.fillna(0), tru_neg.fillna(0)])
    comby = merged.TP + merged.FN + merged.FP + merged.TN

    if graph_out:
        uniq_vals = np.unique(comby.values[~np.isnan(comby.values)]).astype(int)
        colors = [colpac[i] for i in uniq_vals if i < len(colpac)]
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(9, 12))
        fig.suptitle("Comparison of Product and Ground Truth")
        plt.tight_layout(pad=2.5, w_pad=2.0, h_pad=3.5)
        prod_array.plot(ax=axes[0, 0], add_colorbar=False)
        masked_ocean.plot(ax=axes[0, 1], add_colorbar=False)
        temp_mask.plot(ax=axes[1, 0], add_colorbar=False)
        mask.plot(ax=axes[1, 1], add_colorbar=False)
        comby.plot(ax=axes[2, 0], levels=[0.5, 1.5, 2.5, 3.5, 4.5],
                   colors=colors, add_colorbar=False)
        axes[2, 1].axis("off")
        axes[2, 1].text(0.0, 0.5, print_string, fontsize=16,
                        verticalalignment="center", horizontalalignment="left")
        axes[0, 0].set_title("Full Product: Blue=No Burn, Yellow=Burn")
        axes[0, 1].set_title("Ocean mask applied")
        axes[1, 0].set_title("Ground Truth area")
        axes[1, 1].set_title("Ground Truth mask")
        axes[2, 0].set_title("Combined result")
        plt.show()

    return comby, tp, fn, fp, tn


def get_tifs(save_folder: str, suffix: str) -> List[str]:
    """
    Retrieve sorted list of TIFF files with a specific suffix.

    Args:
        save_folder (str): Directory containing TIFF files.
        suffix (str): Suffix to filter files.

    Returns:
        List[str]: Sorted list of TIFF file paths.
    """
    directory = os.path.join(save_folder, "")
    tifs = [os.path.join(directory, file) for root, dirs, files in os.walk(directory)
            for file in sorted(files) if file.endswith(f"{suffix}.tif")]
    return sorted(tifs)


def extract_xy(path: str) -> Optional[Tuple[str, str]]:
    """
    Extract x and y values from a path using pattern 'x<number>y<number>'.

    Args:
        path (str): File path string.

    Returns:
        Optional[Tuple[str, str]]: (x_value, y_value) if pattern found; otherwise, None.
    """
    match = re.search(r"(x\d+y\d+)", path)
    if match:
        extracted = match.group(0)
        matchx = re.search(r"x\d+", extracted)
        matchy = re.search(r"y\d+", extracted)
        if matchx and matchy:
            return matchx.group(0), matchy.group(0)
    print(f"No match for x, y found in {path}")
    return None


def safe_division(numerator: float, denominator: float) -> float:
    """
    Safely divide two numbers.

    Args:
        numerator (float): Numerator.
        denominator (float): Denominator.

    Returns:
        float: Division result or NaN if denominator is zero.
    """
    return numerator / denominator if denominator != 0 else float("nan")


def calculate_classification_metrics(tp: float, tn: float, fp: float, fn: float,
                                     metrics: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Calculate various classification metrics.

    Args:
        tp (float): True positives.
        tn (float): True negatives.
        fp (float): False positives.
        fn (float): False negatives.
        metrics (Optional[List[str]], optional): Specific metrics to calculate. Defaults to all.

    Returns:
        Dict[str, float]: Dictionary of calculated metrics.

    Raises:
        TypeError: If inputs are non-numeric.
        ValueError: If inputs are negative.
    """
    if not all(isinstance(val, (int, float)) for val in [tp, tn, fp, fn]):
        raise TypeError("TP, TN, FP, and FN should be numeric values.")
    if any(val < 0 for val in [tp, tn, fp, fn]):
        raise ValueError("TP, TN, FP, and FN must be non-negative.")

    total = tp + tn + fp + fn
    accuracy = safe_division(tp + tn, total)
    recall = safe_division(tp, tp + fn)
    specificity = safe_division(tn, tn + fp)
    balanced_accuracy = 0.5 * (recall + specificity)
    precision = safe_division(tp, tp + fp)
    npv = safe_division(tn, tn + fn)
    fpr = safe_division(fp, tn + fp)
    fnr = safe_division(fn, tp + fn)
    cohen_den = (tp + fp) * (fp + tn) * (tp + fn) * (fn + tn)
    cohen_kappa = safe_division(2 * (tp * tn - fp * fn), cohen_den)
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = safe_division((tp * tn - fp * fn), mcc_den)
    f1_score = safe_division(2 * tp, 2 * tp + fp + fn)

    available_metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "negative_predictive_value": npv,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "cohen_kappa": cohen_kappa,
        "matthews_correlation_coefficient": mcc,
        "f1_score": f1_score,
    }

    if not metrics:
        return available_metrics
    else:
        results = {}
        for metric in metrics:
            if metric in available_metrics:
                results[metric] = available_metrics[metric]
            else:
                print(f"Metric '{metric}' is not available. Available metrics: {', '.join(available_metrics.keys())}")
        return results


def validation_climate_analysis(fname: str, climate_zone_file: str,
                                climate_legend_fname: str, loc_name: str,
                                vali_year: str) -> pd.DataFrame:
    """
    Perform climate analysis using Koppen-Geiger climate data.

    Args:
        fname (str): Filepath of the validation GeoTIFF.
        climate_zone_file (str): Filepath of the climate zone shapefile.
        climate_legend_fname (str): Filepath of the climate legend text file.
        loc_name (str): Location name for plotting.
        vali_year (str): Validation year string.

    Returns:
        pd.DataFrame: DataFrame with climate class results including area percentage, precision, and recall.
    """
    with rasterio.open(fname) as src:
        metadata = src.meta
        validation_raster = src.read()
        is_valid = (validation_raster != 0).astype(np.uint8)
        raster_polygons = [shape(coords) for coords, value in features.shapes(
            is_valid, transform=src.transform) if value != 0]

    czsf, legend_df = koppen_import(climate_zone_file, climate_legend_fname)
    czsf = czsf.to_crs(epsg=3577)
    raster_poly = gpd.GeoDataFrame(geometry=raster_polygons, crs="EPSG:3577")
    czsf_clip = gpd.overlay(raster_poly, czsf, how="intersection")
    czsf_clip = czsf_clip.drop(columns=["Shape_Area"], errors="ignore")
    czsf_clip["Shape_Area"] = czsf_clip.geometry.area

    fig, ax = plt.subplots(figsize=(10, 10))
    czsf_clip.plot(ax=ax, color=czsf_clip["Color"], legend=True)
    ax.set_title(f"Koppen-Geiger Climate Classification Map of {loc_name}")
    handles = [mpatches.Patch(color=color) for color in czsf_clip["Color"].unique()]
    plt.legend(handles, czsf_clip["Name"].unique().tolist(), loc="lower right")
    save_folder = os.path.dirname(fname)
    save_name = os.path.splitext(os.path.basename(fname))[0]
    plt.savefig(os.path.join(save_folder, f"{save_name}_climate_classification_map.png"))
    plt.show()

    counts_dict: Dict[int, Dict[Any, int]] = {}
    for idx, row in czsf_clip.iterrows():
        mask = features.geometry_mask([row.geometry],
                                      out_shape=validation_raster.shape[-2:],
                                      transform=metadata["transform"],
                                      invert=True)
        values, counts = np.unique(validation_raster[0, :, :][mask], return_counts=True)
        counts_dict[idx] = dict(zip(values, counts))

    for idx, row in czsf_clip.iterrows():
        for value in [1, 2, 3, 4]:
            col_name = str(value)
            czsf_clip.at[idx, col_name] = counts_dict.get(idx, {}).get(value, 0)

    df = czsf_clip.groupby("gridcode")[["1", "2", "3", "4", "Shape_Area"]].sum().rename(
        columns={"1": "TP", "2": "FN", "3": "FP", "4": "TN"}
    )
    df["Precision"] = df.apply(
        lambda row: row["TP"] / (row["TP"] + row["FP"])
        if (row["TP"] + row["FP"]) > 0 else None, axis=1)
    df["Recall"] = df.apply(
        lambda row: row["TP"] / (row["TP"] + row["FN"])
        if (row["TP"] + row["FN"]) > 0 else None, axis=1)

    for idx, row in df.iterrows():
        climate_name = legend_df.loc[legend_df["gridcode"] == idx, "Name"].values[0]
        if row["Precision"] is not None:
            print(f"\nPrecision is {round(row['Precision'] * 100, 2)}% for Climate Class {climate_name}")
        else:
            print(f"\nPrecision is undefined for Climate Class {climate_name}")
        if row["Recall"] is not None:
            print(f"\nRecall is {round(row['Recall'] * 100, 2)}% for Climate Class {climate_name}")
        else:
            print(f"\nRecall is undefined for Climate Class {climate_name}")

    df = df.dropna(subset=["Precision", "Recall"])
    total_area = df["Shape_Area"].sum()
    df["Area_Percent"] = df["Shape_Area"].apply(lambda x: x / total_area)
    results_df = df.drop(columns=["TP", "FN", "FP", "TN", "gridcode", "Color", "Shape_Area"], errors="ignore")
    results_df = results_df.reindex(columns=["Name", "Description", "Area_Percent", "Precision", "Recall"])
    return results_df
