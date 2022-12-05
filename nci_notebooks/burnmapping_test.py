import pandas as pd
import sys
import time
import pyproj
from BurnCube import BurnCube #including burn mapping main functions
bc = BurnCube()

########only for lat lon as input coordinates############
wgs84 = pyproj.Proj(init='epsg:4326')
gda94 = pyproj.Proj(init='epsg:3577')

sites = pd.read_csv('sites.txt')
testsite = int(float(sys.argv[1]))
easting,northing = pyproj.transform(wgs84,gda94,sites.centre_longitude[testsite],sites.centre_latitude[testsite])

#lon = (sites.centre_longitude[testsite]-0.125,sites.centre_longitude[testsite]+0.125) # min to max
#lat = (sites.centre_latitude[testsite]-0.125,sites.centre_latitude[testsite]+0.125)
x = (easting-12500,easting+12500) #25km tile with centre coordinates
y = (northing-12500,northing+12500)
########################################################

#config the data period and mapping period
if sites.year[testsite]>=2013:
    period = ('2013-01-01',str(sites.year[testsite]-1)+'-12-31')# period used for the calculation of geometric median
    sensor = 8
    datatime = ('2013-01-01',str(sites.year[testsite])+'-12-31')
else:
    period = (str(sites.year[testsite]-4)+'-01-01',str(sites.year[testsite]-1)+'-12-31')
    sensor = 5
    datatime = (str(sites.year[testsite]-4)+'-01-01',str(sites.year[testsite])+'-12-31')
res = (-25,25) # resolution in meters
mappingperiod =(str(sites.year[testsite])+'-01-01',str(sites.year[testsite])+'-12-31') # period of interest for change/severity mapping
method = "NBR" # change detection method

#record the computation time for each step

#step1: load data and filtering
start_time = time.monotonic()
bc.load_cube(x, y, res, datatime, [8])
print("---{} minutes for loading data.---".format((time.monotonic()-start_time)/60))
#step2: calculate geometric median
start_time = time.monotonic()
bc.geomedian(period, n_procs=4)
print("---{} minutes for geomedian calculation.---".format((time.monotonic()-start_time)/60))
#step3: calculate cosine distance and nbr distance for reference period
start_time = time.monotonic()
bc.distances(period, n_procs=4)
print("---{} minutes for cos dist.---".format((time.monotonic()-start_time)/60))
#step4: determine the threshold values
start_time = time.monotonic()
bc.outliers()
print("---{} minutes for outliers calculation.---".format((time.monotonic()-start_time)/60))
#step5: calculate the distances to the reference
start_time = time.monotonic()
bc.distances(mappingperiod,n_procs=4)
#step6: burn mapping for the given period
start_time = time.monotonic()
out = bc.severitymapping(mappingperiod, n_procs=4,method=method,growing=True)
print("---{} minutes for burn scar mapping.---".format((time.monotonic()-start_time)/60))

#save the output
filename = '/g/data/xc0/project/Burn_Mapping/TestSites/BurnMapping_'+sites.region_name[testsite]+'_'+mappingperiod[0][0:4]+'_'+'25m.nc'
out.to_netcdf(filename)

