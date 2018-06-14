"""
This script is a simple workflow and can be used for continental production by giving the geographic extent, reference time,
resolution, landsat sensor number, severity mapping period, and the method.

The geometric medians and distances to the geometric medians will be saved in one netcdf file. The severity and the information of burned pixel 
including startdate, duration, easting and northing will be save in a csv file together with a netcdf file with 2D array of severity.

"""

from loaddea import loaddea #for extracting cloudy free data from datacube
from disttogeomedian import geomedian_and_dists#for calculation of reference data for the change detection, i.e. geometric medians and cosine distance
from changedetection import severity_mapping #for severity mapping
import sys

print("Please provide the geographic extent in x and y, reference period for geometric median calculation, resolution, landsat sensor number, severity mapping period, and the mapping method:\nfor example:\n(146.0852,146.3702), (-41.5985,-41.7314), ('2013-01-01','2017-12-31'),(-100,100),8,('2016-01-01','2016-12-31'),3 ")

lon = sys.argv[0] # min to max
lat = sys.argv[1]
reftime = sys.argv[2] # period used for the calculation of geometric median
res = sys.argv[3] # resolution in meters
sensor = sys.argv[4]
severity_period = sys.argv[5] # period of interest for change/severity mapping
method = sys.argv[6] # change detection method

data = loaddea(x=lon,y=lat,time=reftime,resolution=res,landsat_numbers=[sensor]) #load cloud free data for the given region

print("The result will be saved in the current working directory, modify it if required!\n")

out = geomedian_and_dists(data) # calculate the geometrice medians and the distances
filename = 'GMandDist_'+str(lat[0])+'_'+str(lat[1])+'_'+str(lon[0])+'_'+str(lon[1])+'_'+reftime[0]+'_'+reftime[1]+'_Landsat'+str(sensor)+'_'+str(res[1])+'m.nc'
out.to_netcdf(filename) # save the output in a netcdf

firedata,severity=severity_mapping(out,severity_period,method,plot=False)
filename2 = 'Severity_'+str(lat[0])+'_'+str(lat[1])+'_'+str(lon[0])+'_'+str(lon[1])+'_'+severity_period[0]+'_'+severity_period[1]+'_Landsat'+str(sensor)+'_'+str(res[1])+'m.nc'
firedata.to_csv(filename2[0:-3]+'.csv')
severity.to_netcdf(filename2)
