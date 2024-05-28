# # ARDC_burnt_area_mapping_tools.py

# Import required packages
import os
import re
import time
from datetime import datetime

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

os.environ["AWS_NO_SIGN_REQUEST"] = "yes"

start_time = time.time()


def gen_grid_codes(x_range, y_range):
    """
    Generate a list of grid codes in the 'x##y##' format based on the
    provided x and y ranges.

    Parameters
    ----------
    x_range : tuple
        A tuple representing the minimum and maximum x values of the grid.
        The first element of the tuple should be the minimum x value, and
        the second element should be the maximum x value.
    y_range : tuple
        A tuple representing the minimum and maximum y values of the grid.
        The first element of the tuple should be the minimum y value, and
        the second element should be the maximum y value.

    Returns
    -------
    list
        A list of grid codes in the 'x##y##' format, generated based on the
        provided x and y ranges.

    """
    grid_list = []
    for i in range(x_range[0], x_range[1] + 1):
        for j in range(y_range[0], y_range[1] + 1):
            grid_list.append("x" + str(i) + "y" + str(j))

    return grid_list


def koppen_import(koppen_fname, legend_fname):
    """
    Read the Koppen-Geiger climate zone legend from a text file and extract
    the data into a DataFrame. Load the Koppen-Geiger geopackage as a
    GeoDataFrame and merge it with the legend DataFrame based on the
    gridcode. Scale the colors for use in matplotlib plots.

    Parameters
    ----------
    koppen_fname : str
        The path to the Koppen-Geiger GeoPackage.
    legend_fname : str
        The path to the Koppen-Geiger legend text file.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the climate data from the shapefile,
        merged with the legend information.

    """
    # Read in Koppen-Geiger legend text file
    with open(legend_fname) as f:
        lines = f.readlines()

    # Define regular expression patterns to extract the gridcode, climate zone name, description and color
    pattern = re.compile(r"(\d+):\s+(\w+)\s+(.*)\s+\[(\d+)\s+(\d+)\s+(\d+)\]")

    # Initialize lists to store the extracted data
    codes = []
    names = []
    descriptions = []
    colors = []

    # Loop through the lines of the file and extract the data
    for line in lines[3:]:  # Start from the fourth line
        match = pattern.search(line)
        if match:
            code = int(match.group(1))
            name = match.group(2)
            description = match.group(3)
            color = (int(match.group(4)), int(match.group(5)), int(match.group(6)))
            codes.append(code)
            names.append(name)
            descriptions.append(description)
            colors.append(color)

    # Create a DataFrame from the extracted data
    legend_df = pd.DataFrame(
        {"gridcode": codes, "Name": names, "Description": descriptions, "Color": colors}
    )

    # Scale the values in the color data from 0-255 to 0-1 for use in matplotlib plots
    legend_df["Color"] = legend_df["Color"].apply(lambda x: tuple(v / 255.0 for v in x))

    # Read the Climate file using geopandas, and join with the legend_df
    climate_gdf = gpd.read_file(koppen_fname)
    climate_gdf = pd.merge(climate_gdf, legend_df, on="gridcode")

    return climate_gdf, legend_df


def download_s3_files(bucket_name, path_to_download, save_as=None):
    """
    Download files from an Amazon S3 bucket to a local directory.

    Parameters
    ----------
    bucket_name : str
        The name of the Amazon S3 bucket.
    path_to_download : str
        The path of the file in the S3 bucket to download.
    save_as : str, optional
        The local path and filename to save the downloaded file. If not
        specified, the file will be saved with its original name.

    """
    # Create an S3 client
    client = boto3.client(
        "s3", config=botocore.client.Config(signature_version=botocore.UNSIGNED)
    )

    # Download the file from the S3 bucket
    if save_as:
        client.download_file(bucket_name, path_to_download, save_as)
    else:
        client.download_file(bucket_name, path_to_download, path_to_download)


def ardc_year_calc(year_basis, year, extra_months=0):
    """
    Calculate the required start and end dates for a validation period for use
    in extracting ARDC data in datetime format.

    Parameters
    ----------
    year_basis : str
        Either "FY" (Fiscal Year) or "CY" (Calendar Year). Note that "CY" additionally
        includes the preceding December.
    year : int
        The year to extract data from. For example, for the fiscal year 2020-2021,
        or the calendar year 2020, enter 2020.
    extra_months : int, optional
        Number of extra months to include when year_basis is "CY". Default is 0.

    Returns
    -------
    tuple
        A tuple containing the start date and end date of the validation period
        in the format ("%Y-%m-%d").

    Raises
    ------
    ValueError
        If an invalid year basis is provided. The year basis must be either "FY" or "CY".

    """
    if year_basis not in ["FY", "CY"]:
        raise ValueError("Invalid year basis. Must be 'FY' or 'CY'.")

    if year_basis == "FY":
        start_date = datetime(year=year, month=7, day=1)
        end_date = datetime(year=year + 1, month=6, day=30)
    else:
        if extra_months == 0:
            start_date = datetime(year=year, month=1, day=1)
        else:
            start_date = datetime(year=year - 1, month=13 - extra_months, day=1)
        end_date = datetime(year=year, month=12, day=31)

    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")

    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")

    return (start_date, end_date)


def raster_folder_bbox(save_folder):
    """
    Open every TIFF file in a folder directory and return a polygon bounding box
    that encompasses all the TIFF files, as well as the CRS of the polygons.

    Parameters
    ----------
    save_folder : str
        The folder path where the TIFF files are stored.

    Returns
    -------
    tuple
        A tuple containing the polygon bounding box and the CRS of the polygons.

    """
    min_x, min_y, max_x, max_y = (
        float("inf"),
        float("inf"),
        float("-inf"),
        float("-inf"),
    )
    for filename in os.listdir(save_folder):
        if filename.endswith(".tif"):
            file_path = os.path.join(save_folder, filename)
            with rasterio.open(file_path) as src:
                bounds = src.bounds
                # Update the minimum and maximum coordinates
                min_x = min(min_x, bounds.left)
                min_y = min(min_y, bounds.bottom)
                max_x = max(max_x, bounds.right)
                max_y = max(max_y, bounds.top)
                poly_crs = src.crs

            polygon = Polygon(
                [
                    (min_x, min_y),
                    (max_x, min_y),
                    (max_x, max_y),
                    (min_x, max_y),
                    (min_x, min_y),
                ]
            )

    return polygon, poly_crs


def read_shapes(
    ground_truth_file,
    coast_line_file,
    state_bndry_file=None,
    state="",
    column_filter=None,
    filter_entries=None,
):
    """
    Read shapefiles and perform subsetting based on specified criteria.

    Parameters
    ----------
    ground_truth_file : str
        Path to the Ground Truth File (shapefile or geotiff).
    coast_line_file : str
        Path to the Coast Line Shape File (shapefile).
    state_bndry_file : str or None, optional
        Path to the State Boundary File (shapefile), by default None.
    state : str, optional
        Name of the state for subsetting, by default "".
    column_filter : str or None, optional
        The name of the column in the Ground Truth File to use as a filter criterion, by default None.
    filter_entries : list or None, optional
        List of entries to filter the Ground Truth File based on the specified column, by default None.

    Returns
    -------
    tuple
        A tuple containing the subsetted Ground Truth File (gtf_sub),
        the Coast Line Shape File (clsf), and the State Boundary (StateBndry).

    Notes
    -----
    This function reads in shapefiles and performs subsetting based on the specified criteria.
    The function can handle both shapefiles and GeoTIFF files. The Ground Truth File is
    subsetted based on the provided filter criteria, specified by the column_filter and
    filter_entries parameters. The Coast Line Shape File is read and returned as is. If a
    State name is provided, the State Boundary File is read and subsetted to the specified
    State, then reprojected to the EPSG:3577 coordinate reference system. If State is not
    provided, the StateBndry variable will be an empty string.
    """

    # Create Ground Truth File and subset only the relevant burn event in 2019
    dirpath, ext = os.path.splitext(ground_truth_file)
    if ext == ".gpkg":
        gtf = gpd.read_file(ground_truth_file)
    elif ext == ".tif":
        gtf = rio_slurp_xarray(ground_truth_file)
        if gtf.spatial_ref == 3111:
            gtf = gtf.rio.reproject("EPSG:3577")
    else:
        print("Ground Truth File not a shapefile or geotif")

    try:
        if column_filter in gtf.columns:
            gtf_sub = gtf[gtf.ColumnFilter.isin(filter_entries)]
        else:
            gtf_sub = gtf
    except AttributeError:
        gtf_sub = gtf

    # Read in coast line file
    clsf = gpd.read_file(coast_line_file)

    # Read in State boundary, if necessary
    if state:
        bndry = gpd.read_file(state_bndry_file)
        state_bndry = bndry[bndry.STE_NAME21 == state]
        state_bndry = state_bndry.to_crs("EPSG:3577")
    else:
        state_bndry = ""

    return gtf_sub, clsf, state_bndry


def validation_stats(
    product, gtf_sub, clsf, state_bndry, input_type, colpac, graph_out=True
):
    """
    Perform validation statistics on a product by comparing it with ground truth data.

    Parameters
    ----------
    product : str
        Path to the product file (geotiff or other supported format).
    gtf_sub : geopandas.GeoDataFrame or xarray.DataArray
        Subsetted ground truth data.
    clsf : geopandas.GeoDataFrame
        Coast Line Shape File.
    state_bndry : geopandas.GeoDataFrame or str
        State boundary data or empty string.
    input_type : str
        Type of the input file ('tif' for geotiff or other supported formats).
    colpac : list
        List of colors for plotting.
    graph_out : bool, optional
        Flag indicating whether to generate a comparison graph, by default True.

    Returns
    -------
    tuple
        A tuple containing the combined result data array, true positives (TP),
        false negatives (FN), false positives (FP), and true negatives (TN).

    """
    # Create xarray of product
    if input_type == "tif":
        prod_array = rio_slurp_xarray(product)
    else:
        prod_array = rioxarray.open_rasterio(product).Moderate

    if len(state_bndry) != 0:
        # Mask outside State boundary
        state_mask = xr_rasterize(state_bndry, prod_array)

        # Mask out ocean
        oc_mask = xr_rasterize(clsf, prod_array)

        ocean_mask = np.logical_and(oc_mask, state_mask)
    else:
        # Mask out ocean
        ocean_mask = xr_rasterize(clsf, prod_array)

    masked_ocean = prod_array.where(ocean_mask)

    # Mask out all areas outside of the GTSFsub area
    if isinstance(gtf_sub, pd.DataFrame):
        gtf_sub_array = xr_rasterize(gtf_sub, prod_array)
        temp_mask = masked_ocean.where(gtf_sub_array == 1)
        mask = xr_rasterize(gtf_sub, masked_ocean)
    else:
        temp_mask = gtf_sub.rio.reproject_match(prod_array)
        mask = temp_mask.where(temp_mask == 0, 1)

    tp = int(masked_ocean.where(np.logical_and(masked_ocean == 1, mask == 1)).count())

    # Number of pixels within Ground Truth shapefile that are identified as unburnt (False Negatives)
    fn = int(masked_ocean.where(np.logical_and(masked_ocean == 0, mask == 1)).count())

    # Number of pixels outside of the Ground Truth shapefile that are identified as burnt (False Positives)
    fp = int(masked_ocean.where(np.logical_and(masked_ocean == 1, mask == 0)).count())

    # Number of pixels outside of Ground Truth shapefile that are identified as unburnt (True Negatives)
    tn = int(masked_ocean.where(np.logical_and(masked_ocean == 0, mask == 0)).count())

    # Precision = TP/(TP+FP)
    if tp + fp > 0 and tp > 0:
        prec_str = "\nPrecision = " + str(round(100 * tp / (tp + fp), 1)) + "%"
    else:
        prec_str = "\nPrecision is undefined"

    # Recall = TP/(FN+TP)
    if fn + tp > 0 and tp > 0:
        rec_str = "\nRecall = " + str(round(100 * tp / (fn + tp), 1)) + "%"
    else:
        rec_str = "\nRecall is undefined"

    print_string = (
        "True Positives = "
        + str(tp)
        + "\nTrue Negatives = "
        + str(tn)
        + "\nFalse Positives = "
        + str(fp)
        + "\nFalse Negatives = "
        + str(fn)
        + "\n"
        + prec_str
        + rec_str
    )

    tru_pos = masked_ocean.where(np.logical_and(masked_ocean == 1, mask == 1))
    fal_neg = (masked_ocean.where(np.logical_and(masked_ocean == 0, mask == 1)) + 1) * 2
    fal_pos = masked_ocean.where(np.logical_and(masked_ocean == 1, mask == 0)) * 3
    tru_neg = (masked_ocean.where(np.logical_and(masked_ocean == 0, mask == 0)) + 1) * 4
    tru_pos.name = "TP"
    fal_neg.name = "FN"
    fal_pos.name = "FP"
    tru_neg.name = "TN"
    meggy = xr.merge(
        [tru_pos.fillna(0), fal_neg.fillna(0), fal_pos.fillna(0), tru_neg.fillna(0)]
    )
    comby = meggy.TP + meggy.FN + meggy.FP + meggy.TN

    if graph_out:
        #
        # Select only those colours that are represented in the data
        #

        # select unique values
        uniq_vals = np.unique(comby)

        # Remove nan from unique values and convert remaining floats to ints
        colnums = uniq_vals[~np.isnan(uniq_vals)].astype(int)

        # Select only colours that correspond to data values in array
        colpac = [colpac[i] for i in colnums]

        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(9, 12))
        fig.suptitle("Comparison of Product and Ground Truth")
        plt.tight_layout(pad=2.5, w_pad=2.0, h_pad=3.5)
        prod_array.plot(ax=axes[0, 0], add_colorbar=False)
        masked_ocean.plot(ax=axes[0, 1], add_colorbar=False)
        temp_mask.plot(ax=axes[1, 0], add_colorbar=False)
        mask.plot(ax=axes[1, 1], add_colorbar=False)

        comby.plot(
            ax=axes[2, 0],
            levels=[0.5, 1.5, 2.5, 3.5, 4.5],
            colors=colpac,
            add_colorbar=False,
        )
        axes[2, 1].axis("off")
        axes[0, 0].set_title("        Full Product: Blue=No Burn, Yellow=Burn")
        axes[0, 1].set_title("Ocean mask applied")
        axes[1, 0].set_title("Only Ground Truth area shown")
        axes[1, 1].set_title("Only Ground Truth area mask shown")
        axes[2, 0].set_title("Combined result")
        axes[2, 1].text(
            0.0,
            0.5,
            print_string,
            horizontalalignment="left",
            verticalalignment="center",
            fontsize=16,
        )

    return comby, tp, fn, fp, tn


def get_tifs(save_folder, suffix):
    """
    Retrieve a list of file paths for TIFF files in a specified directory
    with a given suffix.

    Parameters
    ----------
    save_folder : str
        The directory path where the TIFF files are located.
    suffix : str
        The suffix that the TIFF files should have. Only files with this
        suffix will be included in the returned list.

    Returns
    -------
    list
        A sorted list of file paths for TIFF files in the specified directory
        that match the given suffix.
    """

    tifs = []
    directory = save_folder + "/"
    for root, dirs, files in sorted(os.walk(directory)):
        for file in files:
            if file.endswith(suffix + ".tif"):
                tifs.append(directory + file)
    return sorted(tifs)


def extract_xy(path):
    """
    Extract the x and y values from a given path using regular expressions.

    The path should contain a substring in the format 'x{number}y{number}'.
    The function will search for this pattern and extract the x and y values
    separately.

    Parameters
    ----------
    path : str
        The path string from which to extract the x and y values.

    Returns
    -------
    tuple
        A tuple containing the extracted x and y values as separate strings.
        If the pattern is not found in the path, None is returned.

    Raises
    ------
    None

    """
    match = re.search(r"x\d+y\d+", path)
    if match:
        extracted_string = match.group(0)
        matchx = re.search(r"x\d+", extracted_string)
        matchy = re.search(r"y\d+", extracted_string)
        extractx = matchx.group(0)
        extracty = matchy.group(0)
        return extractx, extracty
    else:
        print("No match for x, y found in", path)


def calculate_classification_metrics(tp, tn, fp, fn, metrics=[]):
    """
    Calculate classification metrics based on the provided TP, TN, FP, and FN values.

    Parameters
    ----------
    tp : int or float
        Number of true positives.
    tn : int or float
        Number of true negatives.
    fp : int or float
        Number of false positives.
    fn : int or float
        Number of false negatives.
    metrics : list, optional
        List of metrics to calculate. If not provided, all available metrics will be calculated. (Default value = [])

    Returns
    -------
    dict
        A dictionary containing the calculated metrics.

    Raises
    ------
    TypeError
        If TP, TN, FP, or FN are not numeric values.
    ValueError
        If TP, TN, FP, or FN are not positive values greater than or equals to 0.

    """
    if not all(isinstance(val, (int, float)) for val in [tp, tn, fp, fn]):
        raise TypeError("TP, TN, FP, and FN should be numeric values.")

    if not all(val >= 0 for val in [tp, tn, fp, fn]):
        raise ValueError("TP, TN, FP, and FN should be postive values greater than or equals to 0.")

    print("tp, tn, fp, fn", tp, tn, fp, fn)

    available_metrics = {
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "balanced-accuracy": 0.5 * ((tp / (tp + fn)) + (tn / (tn + fp))),
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "negative-predictive-value": tn / (tn + fn),
        "false-positive-rate": fp / (fp + tn),
        "false-negative-rate": fn / (tp + fn),
        "cohen-kappa": (2 * (tp * tn - fp * fn))
        / ((tp + fp) * (fp + tn) * (tp + fn) * (fn + tn)),
        #"g-measure": 2
        #* ((tp / (tp + fp)) * (tp / (tp + fn)))
        #/ ((tp / (tp + fp)) + (tp / (tp + fn))),
        "matthews-correlation-coefficient": ((tp * tn) - (fp * fn))
        / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5,
        "f1-score": (2 * tp) / (2 * tp + fp + fn),
    }

    results = {}

    if not metrics:
        return available_metrics

    for metric in metrics:
        if metric in available_metrics:
            results[metric] = available_metrics[metric]
        else:
            available_metric_names = list(available_metrics.keys())
            print(
                f"Metric '{metric}' is not available. Available metrics: {', '.join(available_metric_names)}"
            )

    return results


def validation_climate_analysis(
    fname, climate_zone_file, climate_legend_fname, loc_name, vali_year
):
    """
    Perform climate analysis using Koppen-Geiger climate data and calculate precision and recall for each climate class.

    Parameters:
        fname (str): Filepath of the validation raster in GeoTIFF format.
        climate_zone_file (str): Filepath of the Climate Zone shapefile in GeoJSON format.
        climate_legend_fname (str): Filepath of the Climate Zone legend data in CSV format.
        loc_name (str): Name of the location for plotting purposes.
        vali_year (str): String of Validation Year and year basis
    Returns:
        pandas.DataFrame: A DataFrame containing the results of climate analysis, including climate class names,
        description, area percentage, precision, and recall.

    The function reads the validation raster data and creates a mask to extract polygons representing valid data.
    It reads the Climate Zone shapefile and clips the polygons based on their intersection with the validation data.
    After calculating the counts for each climate class, it computes precision and recall values for each class.
    The final results are returned as a DataFrame with area percentages, precision, and recall for each climate class.
    The function also saves the plot of the climate classification map as a PNG image in the current working directory.
    """
    # Use rasterio to read in the validation raster. Create variables for the transform.
    # Then create a mask for nodata values, and loop through each feature to create a polygon object of data coverage

    with rasterio.open(fname) as src:
        metadata = src.meta
        validation_raster = src.read()
        # transform = src.transform
        is_valid = (validation_raster != 0).astype(np.uint8)
        raster_polygons = []
        for coords, value in features.shapes(is_valid, transform=src.transform):
            # ignore polygons corresponding to nodata
            if value != 0:
                # convert geojson to shapely geometry
                geom = shape(coords)
                raster_polygons.append(geom)

    # Call the koppen_import function to create a dataframe with the koppen legend data joined.
    czsf, legend_df = koppen_import(climate_zone_file, climate_legend_fname)

    # Set crs for CZSF gdf
    czsf = czsf.to_crs(3577)

    # Convert to polygon object into a gdf
    raster_poly = gpd.GeoDataFrame(crs="epsg:3577", geometry=raster_polygons)

    # Clip polygons from the Climate Zone shapefile that intersect this polygon
    czsf_clip = gpd.overlay(raster_poly, czsf, how="intersection")

    # remove the 'Shape_Area' column and recalculate to adjust for polygons whos area has been clipped
    czsf_clip = czsf_clip.drop(columns="Shape_Area")
    czsf_clip["Shape_Area"] = czsf_clip.geometry.area

    # Plot Climate geodataframe with the climate gridcode symbolised.
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    czsf_clip.plot(ax=ax, color=czsf_clip["Color"], legend=True)
    ax.set_title(f"Koppen-Geiger Climate Classification Map of {loc_name}")
    handles = [mpatches.Patch(color=color) for color in czsf_clip["Color"].unique()]
    plt.legend(handles, czsf_clip["Name"].unique().tolist(), loc="lower right")

    # Save the plot as a PNG image. First calculate the folder and raster name frmo the fname variable.
    save_folder = "/".join(fname.split("/")[:-1])
    save_name = fname.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    plt.savefig(f"{save_folder}/{save_name}_climate_classification_map.png")

    # Display the plot
    plt.show()

    # Create a dictionary to store the counts for each categorical value for each polygon
    counts_dict = {}

    # Loop through the rows of the gdf
    for index, row in czsf_clip.iterrows():
        # Create a mask for the polygon being looped through
        mask = rasterio.features.geometry_mask(
            [row.geometry],
            out_shape=validation_raster.shape[-2:],
            transform=metadata["transform"],
            invert=True,
        )

        # Flatten the array and get the unique values and their counts using np.unique
        values, counts = np.unique(validation_raster[0, :, :][mask], return_counts=True)

        # Add the counts to the dictionary with the index of the current row as the key
        counts_dict[index] = dict(zip(values, counts))

    # Add the counts data to the gdf
    for index, row in czsf_clip.iterrows():
        # Get the counts dictionary for the current row
        counts = counts_dict[index]

        # Loop through the categorical values and add a new column for each value with its count
        for value in [1, 2, 3, 4]:
            column_name = f"{value}"
            if value in counts:
                count = counts[value]
            else:
                count = 0
            czsf_clip.at[index, column_name] = count

    # Group by each unique gridcode and sum count statistics
    df = czsf_clip.groupby("gridcode")[["1", "2", "3", "4", "Shape_Area"]].sum()

    # Rename df columns to their respecive validation result
    df = df.rename(columns={"1": "TP", "2": "FN", "3": "FP", "4": "TN"})

    # Create columns for Precision and Recall, and set values to None
    df["Precision"] = None
    df["Recall"] = None

    # Update df with Climate class legend data
    df = pd.merge(df, legend_df, on="gridcode")

    # For each climate class, calculate Precision and Recall with handling for 0 values
    # Precision = TP/(TP+FP)
    for index, row in df.iterrows():
        if row["TP"] + row["FP"] > 0 and row["TP"] > 0:
            df.loc[index, "Precision"] = row["TP"] / (row["TP"] + row["FP"])
            print(
                "\nPrecision is {}% for Climate Class {}".format(
                    round(df.loc[index, "Precision"] * 100, 2), row["Name"]
                )
            )
        else:
            print("\nPrecision is undefined for Climate Class {}".format(row["Name"]))

        # Recall = TP/(FN+TP)
        if row["FN"] + row["TP"] > 0 and row["TP"] > 0:
            df.loc[index, "Recall"] = row["TP"] / (row["FN"] + row["TP"])
            print(
                "\nRecall is {}% for Climate Class {}".format(
                    round(df.loc[index, "Recall"] * 100, 2), row["Name"]
                )
            )

        else:
            print("\nRecall is undefined for Climate Class {}".format(row["Name"]))

    # drop rows that have nan values for Precision and Recall
    df = df.dropna()

    # Calculate area for each climate class as a percentage of total area
    total_area = df["Shape_Area"].sum()
    df["Area_Percent"] = df["Shape_Area"].apply(lambda x: x / total_area)

    # Create a new results_df and reformat to visualsie the results
    results_df = df.drop(
        ["TP", "FN", "FP", "TN", "gridcode", "Color", "Shape_Area"], axis=1
    )
    results_df = results_df.reindex(
        columns=["Name", "Description", "Area_Percent", "Precision", "Recall"]
    )

    return results_df
