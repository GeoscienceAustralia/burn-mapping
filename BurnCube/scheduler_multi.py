
import os
import datetime
import geopandas as gpd
import subprocess
import sys


def submit_job_to_raijin(tilenumbers,mapyear,method,outdir,subdir,jobfile):
#    logfile = 'logs/{}_{}'.format(
#        datetime.date.today().isoformat(), jobfile.rstrip('.qsub'))
    walltime = 120*len(tilenumbers)
    qsub_call = "qsub -l walltime=%d:00 -v ti=%s,year=%d,method=%s,dir=%s,subdir=%s %s" %(walltime, '_'.join(map(str,tilenumbers)),mapyear,method,outdir,subdir,jobfile)
    try:
        subprocess.call(qsub_call, shell=True)
    except:
        print("qsub error for the statement: %s" %qsub_call)
        
def run_unprocessed_tiles(shpfile,outdir,subdir,mapyear,method,jobfile,tilenumbers,ntile_per_job=24):
    
    toprocess = []
    for tilenumber in tilenumbers:
        
        x0,y0 = shpfile.label[tilenumber].split(',')
        filename = outdir+'BurnMapping_'+str(mapyear)+'_'+x0+'_'+y0+'.nc'
        
        if os.path.isfile(filename):
            print(filename,'processed!')
            continue 
        else:
            print("tile index %d not processed, tile info %s %s" %(tilenumber,x0,y0))
            toprocess.append(tilenumber)
            
    n_jobs = int((len(toprocess)+ntile_per_job-1) / ntile_per_job)
    tilenumber_list = [toprocess[i::n_jobs] for i in range(n_jobs)] 

    for tilenumbers in tilenumber_list:
        submit_job_to_raijin(tilenumbers,mapyear,method,outdir,subdir,jobfile)

if __name__ == '__main__':
    mapyear = 2015
    method = 'NBRdist'

    inputshape = gpd.read_file('kakadu.shp')
    shpfile = gpd.read_file('/g/data/v10/public/firescar/Albers_Grid/Albers_Australia_Coast_Islands_Reefs.shp')
    
    tilenumbers = gpd.sjoin(shpfile,inputshape,op='intersects').index.values
    #the following directories need to be changed before processing
    outputdir = 'output/'
    subdir = 'output/subtiles/'
    jobfile = 'jobs_multi.pbs'
    run_unprocessed_tiles(shpfile,outputdir,subdir,mapyear,method,jobfile,tilenumbers,ntile_per_job=24)
    
