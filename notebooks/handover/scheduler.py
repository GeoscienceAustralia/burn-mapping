
import os
import datetime
import geopandas as gpd
import subprocess


def submit_job_to_raijin(tilenumber,mapyear,method,outdir,jobfile):
    logfile = '/g/data/xc0/user/tian/burn-mapping/logs/{}_{}'.format(
        datetime.date.today().isoformat(), jobfile.rstrip('.qsub'))
    try:
        qsub_call = "qsub -v ti=%d,year=%d,method=%s,dir=%s %s" %(tilenumber,mapyear,method,outdir,jobfile)
        subprocess.call(qsub_call, shell=True)
    except:
        print("qsub error for the statement: %s" %qsub_call)
        
def run_unprocessed_tiles(shpfile,outdir,mapyear,method,jobfile,t0=0,t1=929):
    for tilenumber in range(t0,t1):
    
        x0,y0 = shpfile.label[tilenumber].split(',')
        filename = outdir+'BurnMapping_'+str(mapyear)+'_'+x0+'_'+y0+'.nc'
        #print(filename)
        if os.path.isfile(filename):
            print(filename,'processed!')
            continue 
        else:
            print("tile index %d not processed, tile info %s %s" %(tilenumber,x0,y0))
            submit_job_to_raijin(tilenumber,mapyear,method,outdir,jobfile)

if __name__ == '__main__':
    mapyear = 2018
    method = 'NBRdist'
    shpfile = gpd.read_file('/g/data/v10/public/firescar/Albers_Grid/Albers_Australia_Coast_Islands_Reefs.shp')
    outputdir = '/g/data/xc0/project/Burn_Mapping/continental_100km/'+method+'/'
    jobfile = '/g/data/xc0/user/tian/burn-mapping/jobs.pbs'
    run_unprocessed_tiles(shpfile,outputdir,mapyear,method,jobfile,t0=158,t1=160) #t0,t1 the index number of starting tile and finishing tile

