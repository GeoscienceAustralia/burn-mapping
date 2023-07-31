# # ARDC_burnt_area_mapping_tools.py

# Import required packages
import sys, os

os.environ["AWS_NO_SIGN_REQUEST"] = "yes"
import boto3
import botocore
import geopandas as gpd
import pandas as pd
import re
from datetime import datetime
import rasterio
from shapely.geometry import Polygon

from datetime import datetime
import time
start_time = time.time()

import sys, os, re
import datacube
import geopandas as gpd
import xarray as xr
import pandas as pd
import numpy as np
import rioxarray
import getpass
from datacube.utils.cog import write_cog
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
from matplotlib.patches import Rectangle
from rasterio import features
from shapely.geometry import shape


sys.path.insert(1, "../../Tools/")

from dea_tools.spatial import xr_vectorize, xr_rasterize
from datacube.testutils.io import rio_slurp_xarray


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


def koppen_import(koppen_shp_fname, legend_fname):
    """
    Read the Koppen-Geiger climate zone legend from a text file and extract
    the data into a DataFrame. Load the Koppen-Geiger shapefile as a
    GeoDataFrame and merge it with the legend DataFrame based on the
    gridcode. Scale the colors for use in matplotlib plots.

    Parameters
    ----------
    koppen_shp_fname : str
        The path to the Koppen-Geiger shapefile.
    legend_fname : str
        The path to the Koppen-Geiger legend text file.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the climate data from the shapefile,
        merged with the legend information.

    """
    # Read in Koppen-Geiger legend text file
    with open(legend_fname, "r") as f:
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
    climate_gdf = gpd.read_file(koppen_shp_fname)
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


def ReadShapes(
    GroundTruthFile,
    CoastLineShapeFile,
    StateBndryFile=None,
    State="",
    ColumnFilter=None,
    FilterEntries=None,
):
    """
    Read shapefiles and perform subsetting based on specified criteria.

    Parameters
    ----------
    GroundTruthFile : str
        Path to the Ground Truth File (shapefile or geotiff).
    CoastLineShapeFile : str
        Path to the Coast Line Shape File (shapefile).
    StateBndryFile : str or None, optional
        Path to the State Boundary File (shapefile), by default None.
    State : str, optional
        Name of the state for subsetting, by default "".
    ColumnFilter : str or None, optional
        The name of the column in the Ground Truth File to use as a filter criterion, by default None.
    FilterEntries : list or None, optional
        List of entries to filter the Ground Truth File based on the specified column, by default None.

    Returns
    -------
    tuple
        A tuple containing the subsetted Ground Truth File (GTFsub),
        the Coast Line Shape File (CLSF), and the State Boundary (StateBndry).

    Notes
    -----
    This function reads in shapefiles and performs subsetting based on the specified criteria.
    The function can handle both shapefiles and GeoTIFF files. The Ground Truth File is
    subsetted based on the provided filter criteria, specified by the ColumnFilter and
    FilterEntries parameters. The Coast Line Shape File is read and returned as is. If a
    State name is provided, the State Boundary File is read and subsetted to the specified
    State, then reprojected to the EPSG:3577 coordinate reference system. If State is not
    provided, the StateBndry variable will be an empty string.
    """

    # Create Ground Truth File and subset only the relevant burn event in 2019
    dirpath, ext = os.path.splitext(GroundTruthFile)
    if ext == ".shp":
        GTF = gpd.read_file(GroundTruthFile)
    elif ext == ".tif":
        GTF = rio_slurp_xarray(GroundTruthFile)
        if GTF.spatial_ref == 3111:
            GTF = GTF.rio.reproject("EPSG:3577")
    else:
        print("Ground Truth File not a shapefile or geotif")

    try:
        if ColumnFilter in GTF.columns:
            GTFsub = GTF[GTF.ColumnFilter.isin(FilterEntries)]
        else:
            GTFsub = GTF
    except AttributeError:
        GTFsub = GTF

    # Read in coast line shape file
    CLSF = gpd.read_file(CoastLineShapeFile)

    # Read in State boundary, if necessary
    if State:
        Bndry = gpd.read_file(StateBndryFile)
        StateBndry = Bndry[Bndry.STE_NAME21 == State]
        StateBndry = StateBndry.to_crs("EPSG:3577")
    else:
        StateBndry = ""

    return GTFsub, CLSF, StateBndry


def ValidationStats(
    Product, GTFsub, CLSF, StateBndry, inputType, colpac, GraphOut=True
):
    """
    Perform validation statistics on a product by comparing it with ground truth data.

    Parameters
    ----------
    Product : str
        Path to the product file (geotiff or other supported format).
    GTFsub : geopandas.GeoDataFrame or xarray.DataArray
        Subsetted ground truth data.
    CLSF : geopandas.GeoDataFrame
        Coast Line Shape File.
    StateBndry : geopandas.GeoDataFrame or str
        State boundary data or empty string.
    inputType : str
        Type of the input file ('tif' for geotiff or other supported formats).
    colpac : list
        List of colors for plotting.
    GraphOut : bool, optional
        Flag indicating whether to generate a comparison graph, by default True.

    Returns
    -------
    tuple
        A tuple containing the combined result data array, true positives (TP),
        false negatives (FN), false positives (FP), and true negatives (TN).

    """
    # Create xarray of Product
    if inputType == "tif":
        ProdArray = rio_slurp_xarray(Product)
    else:
        ProdArray = rioxarray.open_rasterio(Product).Moderate

    if len(StateBndry) != 0:
        # Mask outside State boundary
        StateMask = xr_rasterize(StateBndry, ProdArray)

        # Mask out ocean
        OCMask = xr_rasterize(CLSF, ProdArray)

        OceanMask = np.logical_and(OCMask, StateMask)
    else:
        # Mask out ocean
        OceanMask = xr_rasterize(CLSF, ProdArray)

    MaskedOcean = ProdArray.where(OceanMask)

    # Mask out all areas outside of the GTSFsub area
    if isinstance(GTFsub, pd.DataFrame):
        GTFsubArray = xr_rasterize(GTFsub, ProdArray)
        TempMask = MaskedOcean.where(GTFsubArray == 1)
        Mask = xr_rasterize(GTFsub, MaskedOcean)
    else:
        TempMask = GTFsub.rio.reproject_match(ProdArray)
        Mask = TempMask.where(TempMask == 0, 1)

    TP = int(MaskedOcean.where(np.logical_and(MaskedOcean == 1, Mask == 1)).count())

    # Number of pixels within Ground Truth shapefile that are identified as unburnt (False Negatives)
    FN = int(MaskedOcean.where(np.logical_and(MaskedOcean == 0, Mask == 1)).count())

    # Number of pixels outside of the Ground Truth shapefile that are identified as burnt (False Positives)
    FP = int(MaskedOcean.where(np.logical_and(MaskedOcean == 1, Mask == 0)).count())

    # Number of pixels outside of Ground Truth shapefile that are identified as unburnt (True Negatives)
    TN = int(MaskedOcean.where(np.logical_and(MaskedOcean == 0, Mask == 0)).count())

    # Precision = TP/(TP+FP)
    if TP + FP > 0 and TP > 0:
        PrecStr = "\nPrecision = " + str(round(100 * TP / (TP + FP), 1)) + "%"
    else:
        PrecStr = "\nPrecision is undefined"

    # Recall = TP/(FN+TP)
    if FN + TP > 0 and TP > 0:
        RecStr = "\nRecall = " + str(round(100 * TP / (FN + TP), 1)) + "%"
    else:
        RecStr = "\nRecall is undefined"

    PrintString = (
        "True Positives = "
        + str(TP)
        + "\nTrue Negatives = "
        + str(TN)
        + "\nFalse Positives = "
        + str(FP)
        + "\nFalse Negatives = "
        + str(FN)
        + "\n"
        + PrecStr
        + RecStr
    )

    TruPos = MaskedOcean.where(np.logical_and(MaskedOcean == 1, Mask == 1))
    FalNeg = (MaskedOcean.where(np.logical_and(MaskedOcean == 0, Mask == 1)) + 1) * 2
    FalPos = MaskedOcean.where(np.logical_and(MaskedOcean == 1, Mask == 0)) * 3
    TruNeg = (MaskedOcean.where(np.logical_and(MaskedOcean == 0, Mask == 0)) + 1) * 4
    TruPos.name = "TP"
    FalNeg.name = "FN"
    FalPos.name = "FP"
    TruNeg.name = "TN"
    Meggy = xr.merge(
        [TruPos.fillna(0), FalNeg.fillna(0), FalPos.fillna(0), TruNeg.fillna(0)]
    )
    Comby = Meggy.TP + Meggy.FN + Meggy.FP + Meggy.TN

    if GraphOut:
        #
        # Select only those colours that are represented in the data
        #

        # select unique values
        uniqVals = np.unique(Comby)

        # Remove nan from unique values and convert remaining floats to ints
        colnums = uniqVals[~np.isnan(uniqVals)].astype(int)

        # Select only colours that correspond to data values in array
        colpac = [colpac[i] for i in colnums]

        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(9, 12))
        fig.suptitle("Comparison of Product and Ground Truth")
        plt.tight_layout(pad=2.5, w_pad=2.0, h_pad=3.5)
        ProdArray.plot(ax=axes[0, 0], add_colorbar=False)
        MaskedOcean.plot(ax=axes[0, 1], add_colorbar=False)
        TempMask.plot(ax=axes[1, 0], add_colorbar=False)
        Mask.plot(ax=axes[1, 1], add_colorbar=False)

        Comby.plot(
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
            PrintString,
            horizontalalignment="left",
            verticalalignment="center",
            fontsize=16,
        )

    return Comby, TP, FN, FP, TN


def GetTifs(save_folder, suffix):
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

    Tifs = []
    directory = save_folder + "/"
    for root, dirs, files in sorted(os.walk(directory)):
        for file in files:
            if file.endswith(suffix + ".tif"):
                Tifs.append(directory + file)
    return sorted(Tifs)


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


def calculate_classification_metrics(TP, TN, FP, FN, metrics=[]):
    """
    Calculate classification metrics based on the provided TP, TN, FP, and FN values.

    Parameters
    ----------
    TP : int or float
        Number of true positives.
    TN : int or float
        Number of true negatives.
    FP : int or float
        Number of false positives.
    FN : int or float
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
        If TP, TN, FP, or FN are not positive values greater than 0.

    """
    if not all(isinstance(val, (int, float)) for val in [TP, TN, FP, FN]):
        raise TypeError("TP, TN, FP, and FN should be numeric values.")

    if not all(val > 0 for val in [TP, TN, FP, FN]):
        raise ValueError("TP, TN, FP, and FN should be postive values greater than 0.")

    available_metrics = {
        "accuracy": (TP + TN) / (TP + TN + FP + FN),
        "balanced-accuracy": 0.5 * ((TP / (TP + FN)) + (TN / (TN + FP))),
        "precision": TP / (TP + FP),
        "recall": TP / (TP + FN),
        "specificity": TN / (TN + FP),
        "negative-predictive-value": TN / (TN + FN),
        "false-positive-rate": FP / (FP + TN),
        "false-negative-rate": FN / (TP + FN),
        "cohen-kappa": (2 * (TP * TN - FP * FN))
        / ((TP + FP) * (FP + TN) * (TP + FN) * (FN + TN)),
        "g-measure": 2
        * ((TP / (TP + FP)) * (TP / (TP + FN)))
        / ((TP / (TP + FP)) + (TP / (TP + FN))),
        "matthews-correlation-coefficient": ((TP * TN) - (FP * FN))
        / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5,
        "f1-score": (2 * TP) / (2 * TP + FP + FN),
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
    fname, ClimateZoneShapeFile, ClimateLegend_fname, loc_name, vali_year
):
    """
    Perform climate analysis using Koppen-Geiger climate data and calculate precision and recall for each climate class.

    Parameters:
        fname (str): Filepath of the validation raster in GeoTIFF format.
        ClimateZoneShapeFile (str): Filepath of the Climate Zone shapefile in GeoJSON format.
        ClimateLegend_fname (str): Filepath of the Climate Zone legend data in CSV format.
        loc_name (str): Name of the location for plotting purposes.
        vali_year (str): String of Validation Year and year basis
    Returns:
        pandas.DataFrame: A DataFrame containing the results of climate analysis, including climate class names,
        description, area percentage, precision, and recall.

    The function reads the validation raster data and creates a mask to extract polygons representing valid data.
    It then reads the Climate Zone shapefile and clips the polygons based on their intersection with the validation data.
    After calculating the counts for each climate class, it computes precision and recall values for each class.
    The final results are returned as a DataFrame with area percentages, precision, and recall for each climate class.
    The function also saves the plot of the climate classification map as a PNG image in the current working directory.
    """
    # Use rasterio to read in the validation raster. Create variables for the transform.
    # Then create a mask for nodata values, and loop through each feature to create a polygon object of data coverage

    with rasterio.open(fname) as src:
        metadata = src.meta
        validation_raster = src.read()
        transform = src.transform
        is_valid = (validation_raster != 0).astype(np.uint8)
        raster_polygons = []
        for coords, value in features.shapes(is_valid, transform=src.transform):
            # ignore polygons corresponding to nodata
            if value != 0:
                # convert geojson to shapely geometry
                geom = shape(coords)
                raster_polygons.append(geom)

    # Call the koppen_import function to create a dataframe with the koppen legend data joined.
    CZSF, legend_df = koppen_import(ClimateZoneShapeFile, ClimateLegend_fname)

    # Set crs for CZSF gdf
    CZSF = CZSF.to_crs(3577)

    # Convert to polygon object into a gdf
    RasterPoly = gpd.GeoDataFrame(crs="epsg:3577", geometry=raster_polygons)

    # Clip polygons from the Climate Zone shapefile that intersect this polygon
    CZSFclip = gpd.overlay(RasterPoly, CZSF, how="intersection")

    # remove the 'Shape_Area' column and recalculate to adjust for polygons whos area has been clipped
    CZSFclip = CZSFclip.drop(columns="Shape_Area")
    CZSFclip["Shape_Area"] = CZSFclip.geometry.area

    # Plot Climate geodataframe with the climate gridcode symbolised.
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    CZSFclip.plot(ax=ax, color=CZSFclip["Color"], legend=True)
    ax.set_title("Koppen-Geiger Climate Classification Map of {}".format(loc_name))
    handles = [mpatches.Patch(color=color) for color in CZSFclip["Color"].unique()]
    plt.legend(handles, CZSFclip["Name"].unique().tolist(), loc="lower right")

    # Save the plot as a PNG image. First calculate the folder and raster name frmo the fname variable.
    save_folder = fname.split("/")[0]
    save_name = fname.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    plt.savefig("{}/{}_climate_classification_map.png".format(save_folder, save_name))

    # Display the plot
    plt.show()

    # Create a dictionary to store the counts for each categorical value for each polygon
    counts_dict = {}

    # Loop through the rows of the gdf
    for index, row in CZSFclip.iterrows():
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
    for index, row in CZSFclip.iterrows():
        # Get the counts dictionary for the current row
        counts = counts_dict[index]

        # Loop through the categorical values and add a new column for each value with its count
        for value in [1, 2, 3, 4]:
            column_name = f"{value}"
            if value in counts:
                count = counts[value]
            else:
                count = 0
            CZSFclip.at[index, column_name] = count

    # Group by each unique gridcode and sum count statistics
    df = CZSFclip.groupby("gridcode")[["1", "2", "3", "4", "Shape_Area"]].sum()

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
