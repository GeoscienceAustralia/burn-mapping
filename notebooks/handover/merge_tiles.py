import xarray as xr
import geopandas as gpd
from BurnCube import create_attributes
import glob
import argparse

def merge_subtiles(ti,subdir,mapyear,method,outdir):
    shpfile = gpd.read_file('/g/data/v10/public/firescar/Albers_Grid/Albers_Australia_Coast_Islands_Reefs.shp')
    x0,y0 = shpfile.label[ti].split(',')
    filelist = sorted(glob.glob('%s/%s/BurnMapping_%d_%s_%s_*.nc' %(subdir,method,mapyear,x0,y0)))
    if len(filelist)==4:
        print("Merge the following files: %s" %filelist)
        datasets = [xr.open_dataset(file) for file in filelist]
        merged = xr.merge(datasets)
        fname = outdir+'BurnMapping_'+str(mapyear)+'_'+x0+'_'+y0+'.nc'        
        mergeds = merged.copy()
        mergeds['Duration'] = (('y','x'), mergeds.Duration.data.astype('int16'))
        mergeds['Severity'] = (('y','x'), mergeds.Severity.data.astype('float32'))
        mergeds['Severe'] = (('y','x'), mergeds.Severe.data.astype('int16'))
        mergeds['Corroborate'] = (('y','x'), mergeds.Corroborate.data.astype('int16'))
        mergeds['Moderate'] = (('y','x'), mergeds.Moderate.data.astype('int16'))
        mergeds['Cleaned'] = (('y','x'), mergeds.Cleaned.data.astype('int16'))                     
        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in mergeds.data_vars}                       
        ds = create_attributes(mergeds,'Burned Area Map','v1.0', method)
        ds.to_netcdf(fname,encoding=encoding)
        print(fname,'saved!')
    else:
        print("tile process incomplete or no available data for this area %s %s" %(x0,y0))
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="""set up the tile and year information for burn scar mapping for Australia""")
    parser.add_argument('-t', '--tileindex', type=int, required=True, help="index from the shp file")
    parser.add_argument('-m', '--method', type=str, required=True, help="method for mapping i.e. NBR or NBRdist")
    parser.add_argument('-y', '--year', type=int, required=True, help="Year to map [YYYY].")
    parser.add_argument('-i', '--inputdir', type=str, required=True, help="directory of input tiles for merging")
    parser.add_argument('-o', '--outputdir', type=str, required=True, help="directory to save the output")
    args = parser.parse_args()
    merge_subtiles(args.tileindex,args.inputdir,args.year,args.method,args.outputdir)
