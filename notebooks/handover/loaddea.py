
"""
This module load the DEA for the given geographic region and period, and masked out the cloudy area.
Return a xarray with 6 bands reflactance data and one pixel mask used for cloud masking
Required inputs:
x,y in degree
time in string format "%Y-m-d"
landsat_numbers: e.g 5,7,8 or multiple
"""

import xarray as xr
import numpy as np
import warnings
import datacube
from datacube.helpers import ga_pq_fuser
from datacube.storage import masking
warnings.filterwarnings("ignore")

def querydata(x,y,reftime,resolution):
    """
    Define the area of interest:
    x; Defines the spatial region in the x dimension
    y; Defines the spatial region in the y dimension
    time; Defines the temporal extent
    resolution; Defines the spatial resolution
    """
    query = {
        'time': reftime,
        'y': y,
        'x': x,
        'crs':'EPSG:3577',
        'resolution': resolution
    }
    return query

def getLandsatStack(landsat_number,query):
    """
    Extract the Landsat data for the selected region and sensors
    """
    dc = datacube.Datacube(app='TreeMapping.getLandsatStack')
    product= 'ls'+str(landsat_number)+'_nbart_albers'
    rquery = {**query, 
              'resampling' : 'bilinear',
              'measurements' : ['red','green','blue','nir','swir1','swir2']}
    stack = dc.load(product,group_by='solar_day',**rquery) # group by solar day: scenes for same day are merged - causes pixel quality issues
    stack['product'] = ('time', np.repeat(product, stack.time.size)) # adds a label identifying the product
    # now get pixel quality
    qquery = {**query,
              'resampling' : 'nearest',
              'measurements' : ['pixelquality']}
    product= 'ls'+str(landsat_number)+'_pq_albers'
    pq_stack = dc.load(product,group_by='solar_day',fuse_func=ga_pq_fuser,**qquery) # group by solar day: scenes for same day are merged - causes pixel quality issues
    # create land and good quality masks 
    # pandas.DataFrame.from_dict(masking.get_flags_def(pq_stack.pixelquality), orient='index') # to see the list of flags
    pq_stack['land']= masking.make_mask(pq_stack.pixelquality, land_sea='land')
    #pq_stack['ga_good_pixel']= masking.make_mask(pq_stack.pixelquality, ga_good_pixel=True) # not using this as it has issues
    clear_obs= masking.make_mask(pq_stack.pixelquality,cloud_acca='no_cloud')
    clear_obs= clear_obs*masking.make_mask(pq_stack.pixelquality,cloud_fmask='no_cloud')
    clear_obs= clear_obs*masking.make_mask(pq_stack.pixelquality,cloud_shadow_acca='no_cloud_shadow')
    clear_obs= clear_obs*masking.make_mask(pq_stack.pixelquality,cloud_shadow_fmask='no_cloud_shadow')
    pq_stack['no_cloud']=clear_obs
    # align the band and pixel quality stacks 
    # "join=inner" means that images without pixel quality information are rejected.
    lspq_stack, ls_stack = xr.align(pq_stack,stack,join='inner') 
    lspq_stack['good_pixel']= lspq_stack.no_cloud.where(ls_stack.red>0,False,drop=False) # also remove negative reflectances (NaNs)
    return lspq_stack, ls_stack


def clearobsrate(pq_stack):
    """
    Calculate the clear observation coverage at each time step
    """
    pixelquality = pq_stack.pixelquality
    pixelquality.values[pixelquality.values>0]=1
    goodpix = pq_stack.no_cloud*pixelquality*pq_stack.good_pixel # good quality data mask
    NDAYS = len(pq_stack.time)
    clearobs = np.zeros((len(pq_stack.y),len(pq_stack.x)), dtype=np.int16)
    goodcovInd = np.zeros((NDAYS))

    for ti in range(0,NDAYS):
        if (np.nansum(goodpix[ti,:,:])/(len(pq_stack.y)*len(pq_stack.x)))>0.2:
            goodcovInd[ti] = 1
            clearobs = clearobs + goodpix[ti,:,:]
    clearobs = clearobs/NDAYS #percentage of clear observation with more than 20% coverage
    
    
    return clearobs,np.where(goodcovInd==1)[0],goodpix

def loaddea(x,y,reftime,resolution,landsat_numbers):
    #import time
    query = querydata(x,y,reftime,resolution) # query data for the given region and time
    # get landsat data    
    pq_stack = []
    stack = []
    #start = time.monotonic()
    for landsat_number in landsat_numbers:
        lspq_stack, ls_stack = getLandsatStack(landsat_number,query)
        pq_stack.append(lspq_stack)   
        stack.append(ls_stack)   
    pq_stack = xr.concat(pq_stack, dim='time').sortby('time')
    stack = xr.concat(stack, dim='time').sortby('time')
    landmask=pq_stack.land.max(dim='time').values
    pq_stack=pq_stack.drop('land')
    #print("---{} minutes for data loading.---".format((time.monotonic()-start)/60))
    #calculate the clear observation rate at each time step and get a mask for good quality pixel
    start = time.monotonic()
    clearobs,goodcovInd,goodpix = clearobsrate(pq_stack)
    #print("---{} minutes for quality control.---".format((time.monotonic()-start)/60))
    
    #start = time.monotonic()
    
    stack=stack.sel(time=(stack.time[goodcovInd]))
    goodpix=goodpix.sel(time=(goodpix.time[goodcovInd]))
    goodpix=goodpix.where(goodpix==1)


    stack['blue'] = stack.blue*goodpix
    stack['green'] = stack.green*goodpix
    stack['red'] = stack.red*goodpix
    stack['nir'] = stack.nir*goodpix
    stack['swir1'] = stack.swir1*goodpix
    stack['swir2'] = stack.swir2*goodpix


    
    #data = xr.Dataset({'blue':(('time','y','x'),blue[:]),'green':(('time','y','x'),green[:]),'red':(('time','y','x'),red[:]),               'nir':(('time','y','x'),nir[:]),'swir1':(('time','y','x'),swir1[:]),'swir2':(('time','y','x'),swir2[:]),'pixmask':(('time','y','x'),pixmask[:])}, coords={'time':time[:],'y':stack.y[:],'x':stack.x[:]},attrs={'crs':'EPSG:3577'})
    #stack.time.attrs=[]
    #print("---{} minutes for the rest.---".format((time.monotonic()-start)/60))
    return stack




