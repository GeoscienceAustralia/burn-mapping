
"""
This script calculates monthly Geometric Median composites with the observation in 3 months with the current month in centre
Requires the inputs: year, mon e.g. 2015, 1
"""


import xarray as xr
import numpy as np
from geomedian import geometric_median
from datetime import datetime, timedelta
import warnings
import time
import datacube
from datacube.helpers import ga_pq_fuser
from datacube.storage import masking
warnings.filterwarnings("ignore")
import sys


def getLandsatStack(landsat_number,query):
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


#query the 90 days period with the centre of the current day
def querydata(latmin,latmax,lonmin,lonmax,year,mon,day):
    startdate = datetime(year,mon,day)-timedelta(days = 45)
    stopdate = datetime(year,mon,day)+timedelta(days = 45)
    query = {
        'time': (startdate.strftime("%Y-%m-%d"), stopdate.strftime("%Y-%m-%d")),
        'lat': (latmin, latmax),
        'lon': (lonmin, lonmax),
        'measurements' : ['red','green','blue','nir','swir1','swir2'],
        'resolution': (-100, 100)
    }
    return query

latmin = -37.78
latmax = -37.12
lonmin = 144.93
lonmax = 145.96
year = int(sys.argv[1])
mon = int(sys.argv[2])
day = 15
query = querydata(latmin,latmax,lonmin,lonmax,year,mon,day) # query data for the given region and time
print(query)
# get landsat data
landsat_numbers=[5] 
pq_stack = []
stack = []
for landsat_number in landsat_numbers:
    lspq_stack, ls_stack = getLandsatStack(landsat_number,query)
    pq_stack.append(lspq_stack)   
    stack.append(ls_stack)   
pq_stack = xr.concat(pq_stack, dim='time').sortby('time')
stack = xr.concat(stack, dim='time').sortby('time')
landmask=pq_stack.land.max(dim='time').values
pq_stack=pq_stack.drop('land')


def clearobsrate(pq_stack):
    pixelquality = pq_stack.pixelquality
    pixelquality.values[pixelquality.values>0]=1
    goodpix = pq_stack.no_cloud*pixelquality*pq_stack.good_pixel
    NDAYS = len(pq_stack.time)
    clearobs = np.zeros((len(pq_stack.y),len(pq_stack.x)))
    goodcovInd = np.zeros((NDAYS))
    for ti in range(0,NDAYS):
        if (np.nansum(goodpix[ti,:,:])/(len(pq_stack.y)*len(pq_stack.x)))>0.2:
            goodcovInd[ti] = 1
            clearobs = clearobs + goodpix[ti,:,:]
    clearobs = clearobs/NDAYS #percentage of clear observation with more than 20% coverage
    
    #plt.imshow(clearobs)
    #plt.title("Rate of clear observations")
    #plt.colorbar()
    return clearobs,np.where(goodcovInd==1)[0],goodpix

start = time.monotonic()
clearobs,goodcovInd,goodpix = clearobsrate(pq_stack)
row,col = np.where(clearobs>0) 
# calculate geometric median for the clear pixs
NPIX = len(row)
NBANDS = 6
GeoMed = np.zeros((NBANDS,len(pq_stack.y),len(pq_stack.x)))
tol = 1.e-7
MaxIter = 60
del pq_stack
for ni in range(0,NPIX):
    X = np.empty((NBANDS,len(goodcovInd)))
    X[0,:] = stack.blue[goodcovInd,row[ni],col[ni]]*goodpix[goodcovInd,row[ni],col[ni]]
    X[1,:] = stack.green[goodcovInd,row[ni],col[ni]]*goodpix[goodcovInd,row[ni],col[ni]]
    X[2,:] = stack.red[goodcovInd,row[ni],col[ni]]*goodpix[goodcovInd,row[ni],col[ni]]
    X[3,:] = stack.nir[goodcovInd,row[ni],col[ni]]*goodpix[goodcovInd,row[ni],col[ni]]
    X[4,:] = stack.swir1[goodcovInd,row[ni],col[ni]]*goodpix[goodcovInd,row[ni],col[ni]]
    X[5,:] = stack.swir2[goodcovInd,row[ni],col[ni]]*goodpix[goodcovInd,row[ni],col[ni]]
    X[X<=0] = np.nan
    GeoMed[:,row[ni],col[ni]] = geometric_median(X,tol,MaxIter)
end = time.monotonic()
elapsed = end - start
print("Finished with %.1f minutes" % (elapsed/60))       


#save Geometric median to netcdf
datestr=datetime(year,mon,1).strftime("%Y%m")
filename = "Geometric_median_BlackSaturday_VIC_100m_"+datestr+".nc"
ds = xr.Dataset({'blue':(('y','x'),GeoMed[0,:,:]),'green':(('y','x'),GeoMed[1,:,:]),'red':(('y','x'),GeoMed[2,:,:]),               'nir':(('y','x'),GeoMed[3,:,:]),'swir1':(('y','x'),GeoMed[4,:,:]),'swir2':(('y','x'),GeoMed[5,:,:])},               coords={'x':stack.x[:],'y':stack.y[:]},attrs={'geospatial_bounds_crs':'EPSG:4326','lat_min':latmin,                                                        'lat_max':latmax,'lon_min':lonmin,                                                        'lon_max':lonmax})
ds.to_netcdf('/g/data/xc0/project/Burn_Mapping/Geometric_Median/Monthly_GM/'+filename,'w')




    

