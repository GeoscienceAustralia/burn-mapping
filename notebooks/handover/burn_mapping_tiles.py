import os,glob
import time
import geopandas as gpd
import xarray as xr
from BurnCube import BurnCube 
bc = BurnCube()
import argparse
import numpy as np
from BurnCube import create_attributes


def create_empty_dataset(bc,filename):
    out = xr.Dataset(coords={'y': bc.dataset.y[:], 'x': bc.dataset.x[:]})
    out['StartDate'] = (('y', 'x'), np.zeros((len(bc.dataset.y),len(bc.dataset.x)))*np.nan)
    out['Duration'] = (('y', 'x'), np.zeros((len(bc.dataset.y),len(bc.dataset.x))).astype('int16'))    
    out['Severity']=(('y','x'),np.zeros((len(bc.dataset.y),len(bc.dataset.x))).astype('float32'))
    out['Severe'] = (('y', 'x'), np.zeros((len(bc.dataset.y),len(bc.dataset.x))).astype('int16'))
    out['Corroborate'] = (('y', 'x'), np.zeros((len(bc.dataset.y),len(bc.dataset.x))).astype('int16'))
    out['Moderate'] = (('y', 'x'), np.zeros((len(bc.dataset.y),len(bc.dataset.x))).astype('int16'))
    out['Cleaned'] = (('y', 'x'), np.zeros((len(bc.dataset.y),len(bc.dataset.x))).astype('int16'))
    comp = dict(zlib=True, complevel=5) #compression
    encoding = {var: comp for var in out.data_vars}
    ds = create_attributes(out,'Burned Area Map','v1.0', method)
    ds.to_netcdf(fname,encoding=encoding)



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
    _X2 = bc.dataset['cube'].sel(time=slice(mappingperiod[0], mappingperiod[1]))
    if len(_X.time.data)>10 and len(_X2.time.data)>2:
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
            create_empty_dataset(bc,filename)
            return
        else:
            #save the output
            comp = dict(zlib=True, complevel=5) #compression
            encoding = {var: comp for var in out.data_vars}
            out.to_netcdf(filename,encoding=encoding)
            print(filename,' saved.')
    else:
        print("no enough data for the period")
        create_empty_dataset(bc,filename)
        return


# check the existence of tile 
def check_existence(tilenumber,mapyear,method,n_proces,outdir):
    shpfile = gpd.read_file('/g/data/v10/public/firescar/Albers_Grid/Albers_Australia_Coast_Islands_Reefs.shp')
    x0,y0 = shpfile.label[tilenumber].split(',')
    filename = outdir+'BurnMapping_'+str(mapyear)+'_'+x0+'_'+y0+'.nc'
    print(filename)
    if os.path.isfile(filename):
        print(filename,'processed!')
        return
    else:
        subset_process(shpfile,tilenumber,mapyear,method,n_proces,outdir, subset=True)
        
# merge tiles
def merge_tiles(filelist,mapyear,method, x0, y0,outdir):
    
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
        mergeds['Cleaned'] = (('y','x'), mergeds.Moderate.data.astype('int16'))                       
        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in mergeds.data_vars}                       
        ds = create_attributes(mergeds,'Burned Area Map','v1.0', method)
        ds.to_netcdf(fname,encoding=encoding)
        print(fname,'saved!')
    else:
        print("tile process incomplete or no available data for this area %s %s" %(x0,y0))

def subset_process(shpfile,index,mapyear,method,n_proces,outdir,subset=True):
    # process each 100km tile with 4 subtiles at 50km
    x0,y0 = shpfile.label[index].split(',')
    x = (shpfile.loc[index]['X_MIN'], shpfile.loc[index]['X_MAX'])
    y = (shpfile.loc[index]['Y_MIN'], shpfile.loc[index]['Y_MAX'])
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
                burn_mapping(tilex[i-1],tiley[i-1],mapyear,method,n_proces,filename)
        merge_tiles(filelist,mapyear,method, x0, y0,outdir)
    else:
        burn_mapping(x,y,mapyear,method,n_procs,filename)

if __name__ == '__main__':
    #outputdir = '/g/data/xc0/project/Burn_Mapping/continental_100km/'+method+'/' # change here for the correct path
    parser = argparse.ArgumentParser(description="""set up the tile and year information for burn scar mapping for Australia""")
    parser.add_argument('-t', '--tileindex', type=int, required=True, help="index from the shp file")
    parser.add_argument('-m', '--method', type=str, required=True, help="method for mapping i.e. NBR or NBRdist")
    parser.add_argument('-y', '--year', type=int, required=True, help="Year to map [YYYY].")
    parser.add_argument('-np', '--ncpus', type=int, required=True, help="number of cpus to process each tile")
    parser.add_argument('-d', '--dir', type=str, required=True, help="directory to save the output")
    args = parser.parse_args()
    if not os.path.exists(args.dir):
        os.makedirs(args.dir)
    check_existence(tilenumber=args.tileindex,mapyear=args.year,method=args.method, n_proces=args.ncpus,outdir=args.dir)


