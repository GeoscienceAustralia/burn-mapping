import xarray as xr
import glob
import os,sys
from stats import post_filtering
import numpy as np

files = glob.glob('/g/data/v10/public/firescar/BurnScarMap_2016-2017_NBRdist/BurnScarMap_2016-2017*.nc')
outdir = '/g/data/xc0/project/Burn_Mapping/TestSites/continental_100km/filtered_NBRdist_dates/'
start = int(float(sys.argv[1]))
end = start+20

for file in files[start:end]:
    print(start,end)
    if os.path.isfile(outdir+file[58:-3]+'_filtered.nc'):
        print('File exists '+file)
    else:
        sev = xr.open_dataset(file)
        if 'Moderate' in sev.keys():

            BurnPixel = post_filtering(sev,date_filtering=True,hotspots_filtering=False)
           # Mask = BurnPixel.data.copy()
            #sev.Moderate.data = sev.Moderate.data*Mask
            #sev.Severe.data = sev.Severe.data*Mask
            #sev.Severity.data = sev.Severity.data*Mask
            BurnPixel.to_netcdf(outdir+file[58:-3]+'_filtered.nc')
            print(outdir+file[58:-3]+'_filtered.nc')
            del sev
        else:
            BurnArea = np.zeros((sev.Severe.data.shape))
            BurnPixel = xr.Dataset(coords={ 'y': sev.y[:], 'x': sev.x[:]}, attrs={'crs': 'EPSG:3577'})
            BurnPixel['Moderate'] =  (('y', 'x'), BurnArea.astype('int8'))
            BurnPixel.to_netcdf(outdir+file[58:-3]+'_filtered.nc')
