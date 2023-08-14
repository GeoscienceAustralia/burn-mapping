import os
import datacube
import re
import matplotlib.pyplot as plt
import xarray as xr
import geopandas as gpd
import rasterio.features


from dea_tools.datahandling import load_ard
from skimage import morphology 
from scipy import ndimage
from shapely.geometry import shape
from shapely.geometry import Polygon
from datacube.utils.geometry import Geometry, CRS
from datacube.utils.cog import write_cog
from odc.dscache.tools.tiling import parse_gridspec_with_name
from typing import List, Tuple

import argparse

def main():
    parser = argparse.ArgumentParser(description="A script to demonstrate argparse")
    parser.add_argument("regionid", help="The value to be processed")

    args = parser.parse_args()

    #xy = 'x32y15'
    xy = args.regionid

    print("regionid:", xy)

    # the normal config
    odc_config = {"db_hostname": os.getenv("ODC_DB_HOSTNAME"),
                "db_password": os.getenv("ODC_DB_PASSWORD"),
                "db_username": os.getenv("ODC_DB_USERNAME"),
                "db_port": 5432,
                "db_database": os.getenv("ODC_DB_DATABASE")}


    #the special config for our multi year geomedian
    hnrs_config = {"db_hostname": os.getenv("HNRS_DB_HOSTNAME"),
                "db_password": os.getenv("HNRS_DC_DB_PASSWORD"),
                "db_username": os.getenv("HNRS_DC_DB_USERNAME"),
                "db_port": 5432,
                "db_database": os.getenv("HNRS_DC_DB_DATABASE")}


    # this is how we will access the normal data and get our results from the time of the
    # fire and post fire 
    dc = datacube.Datacube(app="post", config=odc_config)
    # this is how we get the multi year geomedian to use as our pre-fire data
    hnrs_dc = datacube.Datacube(app="geomed_loading", config=hnrs_config)

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

        if match:
            x = int(match.group(1))
            y = int(match.group(2))
        else:
            # cannot extract geobox, so we stop here.
            # if we throw exception, it will trigger the Airflow/Argo retry.
            sys.exit(0)

        geobox = gridspec.tile_geobox((x, y))

        # Return the resulting Geometry object
        return datacube.utils.geometry.Geometry(geobox.extent.geom, crs="epsg:3577"), geobox


    #running the polygon function, and getting a usable polygon out
    box = _get_gpgon(xy)
    pgon = box[0]


    # PRE FIRE DATA
    # load in the 4 (financial or calendar) year geomedian
    ds = hnrs_dc.load(product="ga_ls8c_nbart_gm_4cyear_3",
                #x=(136.5, 137.5),
                #y=(-35.6, -36.1),
                geopolygon = pgon,
                time=("2017-01-01", "2017-12-31"),  #calendar year
                #time=("2016-07-01", "2019-06-30"),  #financial year
                output_crs="EPSG:3577",
                dask_chunks={},)

    ds.load()

    # POST FIRE DATA
    #load the post fire data, or the year of interest
    post_ds = load_ard(dc= dc, 
                products=['ga_ls5t_ard_3', 'ga_ls7e_ard_3', 'ga_ls8c_ard_3'],
                #x=(136.5, 137.5),
                #y=(-35.6, -36.1),
                geopolygon = pgon,
                time=("2020-01-01", "2020-12-31"), #calendar year
                #time=("2019-07-01", "2020-06-30"), # financial year
                group_by='solar_day', 
                min_gooddata=0.7,
                output_crs="EPSG:3577",
                dask_chunks={},)

    post_ds.load()


    #load wo to mask out the ocean later
    wofs_summary = dc.load(product="ga_ls_wo_fq_cyear_3",
                #x=(136.5, 137.5),
                #y=(-35.6, -36.1),
                #x=(136.5, 137.),
                #y=(-35.6, -36.1),
                geopolygon = pgon,
                #time=("2019-12-31")) #financial year
                time=("2020")) #calendar year


    # normalised burn ratio
    pre_nbr = (ds.nir - ds.swir2) / (
        ds.nir + ds.swir2
    )

    # bare soil index
    pre_bsi = (
        (ds.swir2 + ds.red)
        - (ds.nir + ds.blue)
    ) / (
        (ds.swir2 + ds.red)
        + (ds.nir + ds.blue)
    )

    # normalised difference vegetation index
    pre_ndvi = (ds.nir - ds.red) / (
        ds.nir + ds.red
    )


    # normalised burn ratio
    post_nbr = (post_ds.nbart_nir - post_ds.nbart_swir_2) / (
        post_ds.nbart_nir + post_ds.nbart_swir_2
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
    post_ndvi = (post_ds.nbart_nir - post_ds.nbart_red) / (
        post_ds.nbart_nir + post_ds.nbart_red
    )


    # delta normalised burn ratio
    delta_nbr = pre_nbr.squeeze("time")-post_nbr


    # delta bare soil index
    delta_bsi = pre_bsi.squeeze("time")-post_bsi


    # delta normalised difference vegetation index
    delta_ndvi = pre_ndvi.squeeze("time")-post_ndvi


    #masking the water and ocean
    wofs_summary_frequency = wofs_summary.frequency
    #wofs_summary_frequency.plot()


    # Create a water mask by identifying areas with water frequency greater than or equal to 0.2
    water_mask = xr.where(wofs_summary_frequency < 0.2, 1., wofs_summary_frequency*0.)
    #water_mask.plot()


    # mask the delta normalised burn ratio
    wo_delta_nbr = water_mask.squeeze("time") * delta_nbr
    #wo_delta_nbr.plot(col="time", col_wrap=2, vmin=-1, vmax=1, cmap="PiYG")

    # mask the delta bare soil index
    wo_delta_bsi = water_mask.squeeze("time") * delta_bsi
    #wo_delta_bsi.plot(col="time", col_wrap=2, vmin=-1, vmax=1, cmap="PiYG")


    # mask the delta normalised difference vegetation index
    wo_delta_ndvi = water_mask.squeeze("time") * delta_ndvi
    #wo_delta_ndvi.plot(col="time", col_wrap=2, vmin=-1, vmax=1, cmap="PiYG")


    # finding the most burnt characteristic for each pixel in each dataset for the time period
    delta_nbr_reduced = wo_delta_nbr.max("time") 
    delta_ndvi_reduced = wo_delta_ndvi.max("time") 
    delta_bsi_reduced = wo_delta_bsi.min("time") 

    # # standardising so all on same negative to positive scale so that very burnt =1
    delta_bsi_reduced = delta_bsi_reduced *-1 # 


    # take the threshold of the various characteristics
    threshold_dbsi = (delta_bsi_reduced >= 0.55 )*1 #Nguyen 2021
    threshold_dnbr = (delta_nbr_reduced >= 0.44 )*1 #USGS
    threshold_dndvi = (delta_ndvi_reduced >= 0.65 )*1 #Szajewska 2018


    # find where the tresholded data agrees, where 2 or more characterisitics indicate
    # fire, we have agreement and cosider the area to be burnt
    stacked_agreement = threshold_dbsi + threshold_dndvi + threshold_dnbr
    stacked_thresholded = stacked_agreement >= 2 


    def dilrode_Delta_dataset(burn_dataset: xr.Dataset)-> xr.DataArray:
        dilated_data = xr.DataArray(morphology.binary_closing(burn_dataset, morphology.disk(3)).astype(burn_dataset.dtype),
                                    coords=burn_dataset.coords)
        erroded_data = xr.DataArray(morphology.erosion(dilated_data, morphology.disk(3)).astype(burn_dataset.dtype),
                                    coords=burn_dataset.coords)
        dilated_data = xr.DataArray(ndimage.binary_dilation(erroded_data, morphology.disk(3)).astype(burn_dataset.dtype),
                                    coords=burn_dataset.coords)
        return dilated_data


    all_burn = dilrode_Delta_dataset(stacked_thresholded)
    #all_burn.plot()


    #build dynamic name
    nm_sensor = "ls" #from dc.load 
    nm_algo = "BAC"
    nm_yeartype = "cy" #decision point here
    nm_collection = "3"
    nm_xy = xy #dynamic build from data loading process
    nm_date = "2020" #see what is in bc, based upon nm_yeartype decision from above
    nm_output = f'{nm_sensor}_{nm_algo}_{nm_yeartype}_{nm_collection}_{nm_xy}_{nm_date}_demo.tif'
    nm_vect = f'{nm_sensor}_{nm_algo}_{nm_yeartype}_{nm_collection}_{nm_xy}_{nm_date}_demo.json'


    #do some crs things that are required to save as vector
    dataset_transform = all_burn.affine

    #print(dataset_transform)

    #dataset_crs = all_burn.rio.crs
    #dataset_transform = all_burn.affine


    # Use rasterio to create features
    vector = rasterio.features.shapes(all_burn.data.astype('float32'),
                                    mask=all_burn.data.astype('float32') == 1,
                                    transform = dataset_transform)
        
    # rasterio.features.shapes outputs tuples. we only want the polygon coordinate portions of the tuples
    vectored_data = list(vector)  # put tuple output in list

    # Extract the polygon coordinates from the list
    polygons = [polygon for polygon, value in vectored_data]

    label='potential_burn'

    # create a list with the data label type
    labels = [label for _ in polygons]

    # Convert polygon coordinates into polygon shapes
    polygons = [shape(polygon) for polygon in polygons]

    # Create a geopandas dataframe populated with the polygon shapes
    data_gdf = gpd.GeoDataFrame(data={'attribute': labels},
                                geometry=polygons,
                                crs=wofs_summary.crs,)

    #save output as GeoJSON
    data_gdf.to_file(nm_vect, driver="GeoJSON")  
    #data_gdf.to_file('allburn2_shp.shp') 


    all_burn.attrs["crs"] = wofs_summary.crs
    all_burn = all_burn.astype('float64')

    write_cog(geo_im=all_burn,
            fname=nm_output,
            overwrite=True,
            nodata=-999)

if __name__ == "__main__":
    main()