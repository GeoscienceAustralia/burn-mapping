
import os
import datetime
import geopandas as gpd
import subprocess
import sys
import argparse

def submit_job_to_raijin(tilenumbers,mapyear,method,outdir,subdir,jobfile):
    """
    Submits the job to gadi
    
    Parameters:
    tilenumbers : array (the labels of the tiles that you want to run)
    mapyear     : int (the year to map)
    method      : str ('NBR' or 'NBRdist')
    outdir      : str (name of the directory to put the output in, inside the working 
                        directory, should exist already)
    subdir      : str (name of the subdirectory (inside outdir) to put the subtiles in,
                      should also exist already)
    jobfile     : str (the job script to use as a template for submission)
    """
    # set the wall_time
    walltime = 120*len(tilenumbers)
    # do the qsub step to submit into the gadi queue    
    qsub_call = "qsub -l walltime=%d:00 -v ti=%s,year=%d,method=%s,dir=%s,subdir=%s %s" %(walltime, '_'.join(map(str,tilenumbers)),mapyear,method,outdir,subdir,jobfile)
    try:
        subprocess.call(qsub_call, shell=True)
    except:
        print("qsub error for the statement: %s" %qsub_call)
        
def run_unprocessed_tiles(shpfile,outdir,subdir,mapyear,method,jobfile,tilenumbers,ntile_per_job=24):
    """
    Checks which tiles haven't been run yet to submit them into the queue and run them
    
    Parameters:
    shpfile       : str (name of shape file that you are using to select your area)
    outdir        : str (name of the directory to put the output in, inside the working 
                        directory, should exist already)
    subdir        : str (name of the subdirectory (inside outdir) to put the subtiles in,
                        should also exist already)
    mapyear       : int (the year to map)
    method        : str ('NBR' or 'NBRdist')
    jobfile       : str (the job script to use as a template for submission)
    tilenumbers   : array (the labels of the tiles that you want to run)
    ntile_per_job : int (set to 24)
    """
    toprocess = []
    for tilenumber in tilenumbers:
        # locate the correct tile to use, and make a filename for it
        x0,y0 = shpfile.label[tilenumber].split(',')
        filename = outdir+'BurnMapping_'+str(mapyear)+'_'+x0+'_'+y0+'.nc'
        # if the file already exists, skip it, but if not, add it to the list to be 
        # processed
        if os.path.isfile(filename):
            print(filename,'processed!')
            continue 
        else:
            print("tile index %d not processed, tile info %s %s" %(tilenumber,x0,y0))
            toprocess.append(tilenumber)
    
    # based on how many tiles have not already been run, set up how many jobs to run, and
    # how many tiles should be done in each job        
    n_jobs = int((len(toprocess)+ntile_per_job-1) / ntile_per_job)
    tilenumber_list = [toprocess[i::n_jobs] for i in range(n_jobs)] 
    # submit a job for each of these
    for tilenumbers in tilenumber_list:
        submit_job_to_raijin(tilenumbers,mapyear,method,outdir,subdir,jobfile)

if __name__ == '__main__':
    # parse in the arguments
    parser = argparse.ArgumentParser(description="""inputs you need to launch the job script for the burn scar mapping""")
    parser.add_argument('-i', '--inputshape', type=str, required=True, help="input shp file")
    #todo is to add an input so that a shape file can be made.
    parser.add_argument('-m', '--method', type=str, required=True, help="method for mapping i.e. NBR or NBRdist")
    parser.add_argument('-y', '--mapyear', type=int, required=True, help="Year to map [YYYY].")
    parser.add_argument('-d', '--outputdir', type=str, required=True, help="directory to save the output")
    parser.add_argument('-sd', '--subdir', type=str, required=True, help="directory to save the subtiles outputdir/subdir")
    parser.add_argument('-j', '--jobfile', type=str, required=True, help="jobfile to use as the template")
    args = parser.parse_args()
    # input area to map, and the Albers shape file
    inputshape = gpd.read_file(args.inputshape)
    shpfile = gpd.read_file('/g/data/v10/public/firescar/Albers_Grid/Albers_Australia_Coast_Islands_Reefs.shp')
    # get the tiles from the intersect of the two
    tilenumbers = gpd.sjoin(shpfile,inputshape,op='intersects').index.values
    run_unprocessed_tiles(shpfile,args.outputdir,args.subdir,args.mapyear,args.method,args.jobfile,tilenumbers,ntile_per_job=24)
    
