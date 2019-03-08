import xarray as xr
import os,glob,sys
import geopandas as gpd

index=int(float(sys.argv[1]))
mapyear=int(float(sys.argv[2]))
method=sys.argv[3]

albers = gpd.read_file('/g/data/v10/public/firescar/Albers_Grid/Albers_Australia_Coast_Islands_Reefs.shp')

x0,y0 = albers.label[index].split(',')
subdir = "/g/data/xc0/project/Burn_Mapping/continental/"+method+"/"
outdir = '/g/data/xc0/project/Burn_Mapping/continental_100km/'+method+"/"
filelist=[]
for i in range(1,5):
    filename = subdir+'BurnMapping_'+str(mapyear)+'_'+x0+'_'+y0+'_tile%d' %i+'_'+method+'.nc'

    if os.path.isfile(filename):
        filelist.append(filename)
    #filelist = glob.glob(subdir+'BurnMapping_'+str(mapyear)+'_'+x0+'_'+y0+'_tile*.nc')
if len(filelist)>0:    
    print(filelist)
    datasets = [xr.open_dataset(file) for file in filelist]
    merged = xr.merge(datasets)
    fname = outdir+'BurnMapping_'+str(mapyear)+'_'+x0+'_'+y0+'.nc'
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in merged.data_vars}
    merged.to_netcdf(fname,encoding=encoding)
    print(fname,'saved!')
else:
    print("tile not processed yet %s %s" %(x0,y0))
