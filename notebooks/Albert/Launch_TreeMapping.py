
# coding: utf-8

import subprocess
import os
import numpy
import time

def make_job_list(lat_bounds:tuple,lon_bounds:tuple,domain_size:float):
    ul_lat_list=list()
    ul_lon_list=list()
    for domain_ul_lat in numpy.arange(numpy.max(lat_bounds),numpy.min(lat_bounds),-domain_size):
        for domain_ul_lon in numpy.arange(numpy.min(lon_bounds),numpy.max(lon_bounds),domain_size):
            ul_lat_list.append(domain_ul_lat)
            ul_lon_list.append(domain_ul_lon)
    return ul_lat_list, ul_lon_list

def qsub(*args: str):
    """Sumbit a PBS jub and return the numeric ID as a string."""
    return subprocess.run(
        args, check=True, stdout=subprocess.PIPE, encoding='utf-8'
    ).stdout.strip().split('.')[0]


# MAIN

# parameters
lat_bounds=(-9,-44)
lon_bounds=(112,154)
#lat_bounds=(-30,-34.5) # for testing
#lon_bounds=(140,144.5) # for testing
domain_size=1.5
secs_between_jobs=20 # build in a wait to try and avoid overloading the DEA license key

# estimate wall time per job
est_walltime_per_degree=0.75 # in hours (based on test runs with 8 cpus)
est_walltime=int(numpy.ceil(est_walltime_per_degree*domain_size**2))
print('Estimated walltime per job '+str(est_walltime)+' hours')

# make joblist
ul_lat_list, ul_lon_list = make_job_list(lat_bounds,lon_bounds,domain_size)
print('Preparing to submit '+str(len(ul_lat_list))+' jobs')

# submit jobs
jobs = dict()
log_dir = '/g/data/xc0/project/GA_burn/temp/logs/'
for di in range(0,len(ul_lat_list)):    
    ul_lat = ul_lat_list[di]
    ul_lon = ul_lon_list[di]
    outfn=f'/g/data/xc0/project/GA_burn/temp/treemap_2017_{ul_lat}S_{ul_lon}E_{domain_size}deg.nc'
    if os.path.isfile(outfn):
        print('File already exists - job not submitted')
    else:
        logfile = f'{log_dir}TreeMapping_{ul_lat}_{ul_lon}'
        jobs[(ul_lat, ul_lon)] = qsub(
                'qsub',
                '-v', f'DOMAIN_UL_LAT={ul_lat},DOMAIN_UL_LON={ul_lon},DOMAIN_SIZE={domain_size}',
                '-l', f'walltime={est_walltime}:00:00',
                '-N', f'TreeMapping_{ul_lat}_{ul_lon}',
                '-o', f'{logfile}.out',
                '-e', f'{logfile}.err',
                'TreeMapping.qsub'
        )
        #print('log file',logfile)
        #job=' '.join(map(str,job)) 
        job_id = jobs[(ul_lat, ul_lon)]
        print(f'Submitted job {job_id} for {ul_lat} deg lat, {ul_lon} deg lon')
        print(f'waiting {secs_between_jobs} seconds before continuing job submission...')
        time.sleep(secs_between_jobs) 


print('==================')
print('All jobs submitted')

