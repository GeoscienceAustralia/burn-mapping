
import os
import datetime
import geopandas as gpd
import subprocess
import sys
import argparse

def submit_job_to_raijin(tilenumbers,mapyear,finyear,method,outdir,subdir,jobfile,project,queue):
    """
    Submits the job to gadi
    
    Parameters:
    tilenumbers : array (the labels of the tiles that you want to run)
    mapyear     : int (the year to map)
    finyear     : boolean (set to true if you want to map july/mapyear to june/mapyear+1)
    method      : str ('NBR' or 'NBRdist')
    outdir      : str (name of the directory to put the output in, inside the working 
                        directory, should exist already)
    subdir      : str (name of the subdirectory (inside outdir) to put the subtiles in,
                      should also exist already)
    jobfile     : str (the job script to use as a template for submission)
    project     : str (your project on gadi)
    queue       : str (the queue you want the job submitted to)
    """
    # set the wall_time
    walltime = 120*len(tilenumbers)
    # do the qsub step to submit into the gadi queue    
    qsub_call = "qsub -P %s -q %s -l walltime=%d:00 -l storage=gdata/%s+gdata/rs0+gdata/v10 -v ti=%s,year=%d,method=%s,dir=%s,subdir=%s,finyear=%s %s" %(project,queue,walltime,project, '_'.join(map(str,tilenumbers)),mapyear,method,outdir,subdir,finyear,jobfile)
    print('The qsub call is:', qsub_call)
    try:
        subprocess.call(qsub_call, shell=True)
    except:
        print("qsub error for the statement: %s" %qsub_call)
        
def run_unprocessed_tiles(shpfile,outdir,subdir,mapyear,finyear,method,jobfile,tilenumbers,project,queue,ntile_per_job=24):
    """
    Checks which tiles haven't been run yet to submit them into the queue and run them
    
    Parameters:
    shpfile       : str (name of shape file that you are using to select your area)
    outdir        : str (name of the directory to put the output in, inside the working 
                        directory, should exist already)
    subdir        : str (name of the subdirectory (inside outdir) to put the subtiles in,
                        should also exist already)
    mapyear       : int (the year to map)
    finyear       : boolean (set to true if you want to map july/mapyear to june/mapyear+1)
    method        : str ('NBR' or 'NBRdist')
    jobfile       : str (the job script to use as a template for submission)
    tilenumbers   : array (the labels of the tiles that you want to run)
    ntile_per_job : int (set to 24)
    project     : str (your project on gadi)
    queue       : str (the queue you want the job submitted to)
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
        submit_job_to_raijin(tilenumbers,mapyear,finyear,method,outdir,subdir,jobfile,project,queue)

if __name__ == '__main__':
    # parse in the arguments
    parser = argparse.ArgumentParser(description="""inputs you need to launch the job script for the burn scar mapping""")
    parser.add_argument('-i', '--inputshape', type=str, required=True, help="input shp file")
    #todo is to add an input so that a shape file can be made.
    parser.add_argument('-m', '--method', type=str, required=True, help="method for mapping i.e. NBR or NBRdist")
    parser.add_argument('-y', '--mapyear', type=int, required=True, help="Year to map [YYYY].")
    parser.add_argument('-fy', '--finyear', type=bool, default=False, help="set to true if you want to map July/mapyear to June/mapyear+1")
    parser.add_argument('-d', '--outputdir', type=str, required=True, help="directory to save the output (no underscores!)")
    parser.add_argument('-sd', '--subdir', type=str, required=True, help="directory to save the subtiles outputdir/subdir (no underscores!)")
    parser.add_argument('-j', '--jobfile', type=str, required=True, help="jobfile to use as the template")
    parser.add_argument('-p', '--project', type=str, required=True, help="project to run on/charge to")
    parser.add_argument('-q', '--queue', type=str, default='normal', help="queue to submit the job to, normal or express")
    args = parser.parse_args()
    # input area to map, and the Albers shape file
    inputshape = gpd.read_file(args.inputshape)
    shpfile = gpd.read_file('/g/data/v10/public/firescar/Albers_Grid/Albers_Australia_Coast_Islands_Reefs.shp')
    # get the tiles from the intersect of the two
    tilenumbers = gpd.sjoin(shpfile,inputshape,op='intersects').index.values
    run_unprocessed_tiles(shpfile,args.outputdir,args.subdir,args.mapyear,args.finyear,args.method,args.jobfile,tilenumbers,args.project,args.queue,ntile_per_job=24)
    
