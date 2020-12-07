import xarray as xr
import numpy as np
import validationtoolbox as val
import matplotlib.pyplot as plt
import geopandas as gpd
import glob
from BurnCube import BurnCube 
bc = BurnCube()
import argparse
from BurnCube import create_attributes
import datacube
from datacube.storage import masking
import sys

"""
This is a script to mask out forests and persistent water from the fire maps.
Masking for water gets rid of oceans or lakes that are tracked as fire.
Masking for forests makes it easier to compare to other methods, which preferentially map forest and can tidy fire maps. 
"""

sys.path.append("/home/100/ekb100/dea-notebooks/Scripts")

dc = datacube.Datacube(app="wofs_mask")

year = 2019
tyear = 2018
in_dir = '2019_east_vic'

tile_list = glob.glob(in_dir+'/BurnMapping_'+str(year)+'_*.nc')

for tile in tile_list:
    # open the tile
    burnmap = xr.open_dataset(tile)
    # make the burn pixel mask
    Burnpixel_mod = val.burnpixel_masking(burnmap,'Moderate') # mask the burnt area with "Medium" burnt area
    Burnpixel_sev = val.burnpixel_masking(burnmap,'Severe')
    # make the tree mask
    mask=val.treecover_masking(year=tyear-1,data=Burnpixel_mod,prctg=60,size='25m')
    # apply the forest mask
    ForestBurnedMod = Burnpixel_mod*mask.ForestMask # burned pixel found in the forest area
    #NoneForestBurnedMod = Burnpixel_mod*mask.NoneForestMask # burned pixel found in the non-forest area
    ForestBurnedSev = Burnpixel_sev*mask.ForestMask # burned pixel found in the forest area
    #NoneForestBurnedSev = Burnpixel_sev*mask.NoneForestMask # burned pixel found in the non-forest area
    # apply the forest mask to all the other layers
    ForestSeverity = mask.ForestMask*burnmap['Severity']
    ForestStartDate = mask.ForestMask*burnmap['StartDate']
    ForestDuration = mask.ForestMask*burnmap['Duration']
    ForestCorroborate = mask.ForestMask*burnmap['Corroborate']
    ForestCleaned = mask.ForestMask*burnmap['Cleaned']
    # make the water mask
    # load the wofs
    wofs = dc.load(product="wofs_annual_summary", like=burnmap, time=str(year))
    # make the mask
    #wofs_mask = xr.where(wofs['frequency'][0,:,:]<0.2, 1, 0)
    wofs_mask = (wofs['frequency'][0,:,:].values<0.2).astype(float)
    # apply the wofs mask
    WOfSModerate = wofs_mask*Burnpixel_mod
    WOfSSevere = wofs_mask*Burnpixel_sev 
    WOfSSeverity = wofs_mask*burnmap['Severity']
    WOfSStartDate = wofs_mask*burnmap['StartDate']
    WOfSDuration = wofs_mask*burnmap['Duration']
    WOfSCorroborate = wofs_mask*burnmap['Corroborate']
    WOfSCleaned = wofs_mask*burnmap['Cleaned']
    # apply the wofs to the forest data mask
    WOfSForestModerate = wofs_mask*ForestBurnedMod
    WOfSForestSevere = wofs_mask*ForestBurnedSev 
    WOfSForestSeverity = wofs_mask*ForestSeverity
    WOfSForestStartDate = wofs_mask*ForestStartDate
    WOfSForestDuration = wofs_mask*ForestDuration
    WOfSForestCorroborate = wofs_mask*ForestCorroborate
    WOfSForestCleaned = wofs_mask*ForestCleaned    
    # save the masked data as netcdfs, along with all the data already in the netcdf
    combined_ds = xr.Dataset({"StartDate":burnmap['StartDate'],
                         "Duration":burnmap['Duration'],
                         "Severity":burnmap['Severity'],
                         "Severe":burnmap['Severe'],
                         "Moderate":burnmap['Moderate'],
                         "Corroborate":burnmap['Corroborate'],
                         "Cleaned":burnmap['Cleaned'],
                         "ForestModerate":ForestBurnedMod,
                         "ForestSevere":ForestBurnedSev,
                         "ForestMask":mask.ForestMask,
                         "ForestSeverity":ForestSeverity,
                         "ForestStartDate":ForestStartDate,
                         "ForestDuration":ForestDuration,
                         "ForestCorroborate":ForestCorroborate,
                         "ForestCleaned":ForestCleaned,
                         "WOfSModerate":WOfSModerate,
                         "WOfSSevere":WOfSSevere,
                         "WOfSSeverity":WOfSSeverity,
                         "WOfSStartDate":WOfSStartDate,
                         "WOfSDuration":WOfSDuration,
                         "WOfSCorroborate":WOfSCorroborate,
                         "WOfSCleaned":WOfSCleaned,
                         "WOfSForestModerate":WOfSForestModerate,
                         "WOfSForestSevere":WOfSForestSevere,
                         "WOfSForestSeverity":WOfSForestSeverity,
                         "WOfSForestStartDate":WOfSForestStartDate,
                         "WOfSForestDuration":WOfSForestDuration,
                         "WOfSForestCorroborate":WOfSForestCorroborate,
                         "WOfSForestCleaned":WOfSForestCleaned
                         })
    new_name = tile.split('.')[0]+'_Forest_WOfS.nc'
    print('saving:', new_name)
    combined_ds.to_netcdf(new_name)
