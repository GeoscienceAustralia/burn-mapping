
import xarray as xr
import geopandas as gpd
import numpy as np
from BurnCube import create_attributes
import os,glob


year = 2016
method = 'NBRdist'
outputdir = '/g/data/xc0/project/Burn_Mapping/continental_100km/%s' %method
os.chdir(outputdir)
filelists = glob.glob('BurnMapping_%d_*' %(year))
newdir = '/g/data/xc0/project/Burn_Mapping/continental_100km/%s/%d/' %(method,year)


for fi,fname in enumerate(filelists):
    if os.path.isfile(fname):
        ds = xr.open_dataset(fname)
        dsnew = ds.copy()    
        dsnew['Duration'] = (('y','x'), dsnew.Duration.data.astype('int16'))
        dsnew['Severity'] = (('y','x'), dsnew.Severity.data.astype('float32'))
        dsnew['Severe'] = (('y','x'), dsnew.Severe.data.astype('int16'))
        dsnew['Corroborate'] = (('y','x'), dsnew.Corroborate.data.astype('int16'))
        dsnew['Moderate'] = (('y','x'), dsnew.Moderate.data.astype('int16'))
        dsnew['Cleaned'] = (('y','x'), dsnew.Moderate.data.astype('int16'))                       
        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in dsnew.data_vars}                       
        dsnew = create_attributes(dsnew,'Burned Area Map','v1.0', method)
        dsnew.to_netcdf(newdir+fname,encoding=encoding)
        dsnew.close()
        ds.close()
        print("reformat finished for %s" %fname)

