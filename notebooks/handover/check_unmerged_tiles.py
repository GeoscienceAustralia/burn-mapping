

import xarray as xr
import geopandas as gpd
import numpy as np
from BurnCube import create_attributes
import os,glob
import argparse
import subprocess


def check_processed_tiles(shpfile,outdir):
    tiles = []
    for tilenumber in range(0,len(shpfile)):    
        x0,y0 = shpfile.label[tilenumber].split(',')
        filename = outdir+'BurnMapping_'+str(mapyear)+'_'+x0+'_'+y0+'.nc'
        if os.path.isfile(filename):            
            continue
        else:
            #print("tile index %d, tile info %s %s" %(tilenumber,x0,y0))
            tiles.append(tilenumber)
    return tiles


if __name__ == '__main__':
    #change here for the mapping period and method
    mapyear = 2017
    method = 'NBR'
    ncpus = 16 # the number of processors can be changed here but need to be change in the merge_tile_job.pbs
    sourcedir = '/g/data/xc0/project/Burn_Mapping/continental_100km/'+method+'/'
    subdir = '/g/data/xc0/project/Burn_Mapping/continental'
    albers = gpd.read_file('/g/data/v10/public/firescar/Albers_Grid/Albers_Australia_Coast_Islands_Reefs.shp')
    indices = check_processed_tiles(albers,sourcedir)
    outdir = '/g/data/xc0/project/Burn_Mapping/continental_100km/%s/%d/' %(method,mapyear)
    pbsdir = '/g/data/xc0/user/tian/burn-mapping' #directory for the bash scripts 
    chunk = len(indices) // ncpus
    
    for i in range(0, len(indices), ncpus):
        
        j = min(len(indices), i + ncpus) 
        f = open("mergejobs.txt" ,'w')       
        for ind in indices[i:j]:
            
            f.write("python3  merge_tiles.py -t %d -y %d -m %s -i %s -o %s \n" %(ind,mapyear,method,subdir,outdir))
        f.close()
        
        
        f = open("%s/merge_tile_job.pbs" %pbsdir,'r+')
        newpbs = f.readlines()
        ncpus = j-i
        newpbs[3] = '#PBS -l ncpus=%d\n' %ncpus
        newpbs[-1] = './conc_exec -f mergejobs.txt -c %d ' %(ncpus)
        f.seek(0)
        f.truncate()
        # re-write the content with the updated content
        f.write(''.join(newpbs))
        # close file
        f.close()
        subprocess.call("qsub %s/merge_tile_job.pbs " %(pbsdir), shell=True)

