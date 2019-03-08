
# coding: utf-8

# In[1]:


import pandas as pd
import sys,os,glob
import time
#import pyproj
import geopandas as gpd
import xarray as xr
from BurnCube import BurnCube #including burn mapping main functions
bc = BurnCube()




def burn_mapping(x,y,mapyear,method,n_procs,filename,res=(-25,25)):
    #config the data period and mapping period
    if mapyear>=2013:
        period = ('2013-01-01',str(mapyear-1)+'-01-01')# period used for the calculation of geometric median
        sensor = 8
        datatime = ('2013-01-01',str(mapyear)+'-12-31')
    else:
        period = (str(mapyear-4)+'-01-01',str(mapyear-1)+'-12-31')
        sensor = 5
        datatime = (str(mapyear-4)+'-01-01',str(mapyear)+'-12-31')

    mappingperiod =(str(mapyear)+'-01-01',str(mapyear)+'-12-31') # period of interest for change/severity mapping
    #record the computation time for each step
    print(x,y,mappingperiod)
    #step1: load data and filtering
    start_time = time.monotonic()
    x = (x[0],x[1])
    y = (y[0],y[1])
    print(x,y)
    try:
        bc.load_cube(x, y, res, datatime, [sensor])
        print("---{} minutes for loading data.---".format((time.monotonic()-start_time)/60))
        
    except:
        print("Problem loading data")
        if 'x' not in bc.dataset.keys() or len(bc.dataset.time)<12:
            print('No data available for the selected extent.')           
            return
    _X = bc.dataset['cube'].sel(time=slice(period[0], period[1]))
    t_dim = _X.time.data
    _X2 = bc.dataset['cube'].sel(time=slice(mappingperiod[0], mappingperiod[1]))
    t_dim2 = _X2.time.data
    if len(t_dim)>6 and len(t_dim2)>2:
        start_time = time.monotonic()
        bc.geomedian(period, n_procs=n_procs)

        print("---{} minutes for geomedian calculation.---".format((time.monotonic()-start_time)/60))
        #step3: calculate cosine distance and nbr distance for reference period
        start_time = time.monotonic()
        bc.distances(period, n_procs=n_procs)
        print("---{} minutes for cos dist.---".format((time.monotonic()-start_time)/60))
        #step4: determine the threshold values
        start_time = time.monotonic()
        bc.outliers()
        print("---{} minutes for outliers calculation.---".format((time.monotonic()-start_time)/60))
        #step5: calculate the distances to the reference
        start_time = time.monotonic()
        bc.distances(mappingperiod,n_procs=n_procs)
        #step6: burn mapping for the given period
        start_time = time.monotonic()
        out = bc.severitymapping(mappingperiod, n_procs,method=method,growing=True)
        print("---{} minutes for burn scar mapping.---".format((time.monotonic()-start_time)/60))
        if out is None:
            print("No data available for mapping")
            return
        else:
            #save the output
            comp = dict(zlib=True, complevel=5)
            encoding = {var: comp for var in out.data_vars}
            out.to_netcdf(filename,encoding=encoding)
            print(filename,' saved.') 
    else:
        print("no enough data for the period")
        return                                           


albers = gpd.read_file('/g/data/v10/public/firescar/Albers_Grid/Albers_Australia_Coast_Islands_Reefs.shp')



subset = True
label = None
#mapyear = 2017
#method = 'NBR'
index = int(float(sys.argv[1]))
mapyear = int(float(sys.argv[2]))
method = sys.argv[3]

n_procs=8
    
outputdir = '/g/data/xc0/project/Burn_Mapping/continental_100km/'+method+'/'
if not os.path.exists(outputdir):
    os.makedirs(outputdir)

# check the existence of tile
def check_existence(tilenumber,shpfile,method,n_proces,outdir):
    x0,y0 = shpfile.label[tilenumber].split(',')
    filename = outdir+'BurnMapping_'+str(mapyear)+'_'+x0+'_'+y0+'.nc'
    print(filename)
    if os.path.isfile(filename):
        print(filename,'processed!')
        return
    else:
        subset_process(shpfile,tilenumber,method,n_proces,outdir, subset=True)

def subset_process(shpfile,index,method,n_proces,outdir,subset=True):
    x0,y0 = albers.label[index].split(',')
    x = (albers.loc[index]['X_MIN'], albers.loc[index]['X_MAX'])
    y = (albers.loc[index]['Y_MIN'], albers.loc[index]['Y_MAX'])
    if subset == True:
        xm, ym = (x[0]+x[1])/2, (y[0]+y[1])/2
        x1, x2 = (x[0], xm), (xm, x[1])
        y1, y2 = (y[0], ym), (ym, y[1]) 
        filelist = []
        tilex = [x1,x1,x2,x2]
        tiley = [y1,y2,y1,y2]
        subdir = '/g/data/xc0/project/Burn_Mapping/continental/'+method+'/'
        for i in range(1,5): 
            filename = subdir+'BurnMapping_'+str(mapyear)+'_'+x0+'_'+y0+'_tile%d' %i+'_'+method+'.nc'

            if os.path.isfile(filename):
                filelist.append(filename)
                print(filename,'processed!')
            else: 
                burn_mapping(tilex[i-1],tiley[i-1],mapyear,method,n_procs,filename)

        #if len(glob.glob(subdir+'BurnMapping_'+str(mapyear)+'_'+x0+'_'+y0+'_tile*.nc'))>0:
        #    print(filelist)
        #    datasets = [xr.open_dataset(file) for file in filelist]
        #    merged = xr.merge(datasets)
        #    fname = outdir+'BurnMapping_'+str(mapyear)+'_'+x0+'_'+y0+'.nc'
        #    comp = dict(zlib=True, complevel=5)
        #    encoding = {var: comp for var in merged.data_vars}
        #    merged.to_netcdf(fname,encoding=encoding)
        #    print(fname,'saved!')
    else:
        burn_mapping(x,y,mapyear,method,n_procs,filename)

check_existence(index,albers,method,n_procs,outputdir)


