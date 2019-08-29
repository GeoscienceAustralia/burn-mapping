
import os, glob
from math import ceil

# split jobs into batches
output_dir = 'batch_jobs'
if not os.path.exists(output_dir): os.mkdir(output_dir)
output_basename = '%s/jobs_tilelist'%output_dir
os.system('rm -f {0}_??'.format(output_basename))
processed_dir = '/g/data/u46/users/fxy120/firescar/BurnScarMap_2016-2017_NBRdist'
ntile_per_job = 16

# list of all tiles
tilesfile = '/g/data/u46/users/fxy120/sensor_data_maps/Tiles_Australia_Coast_Islands_Reefs.txt'
tilesfile = '/g/data/u46/users/fxy120/sensor_data_maps/Tiles_Australia_Coast_and_Islands.txt'
tiles = [l.strip() for l in open(tilesfile).readlines()]

# completed tiles
files=glob.glob('%s/BurnScar*.nc'%processed_dir)
if len(files)>0:
    completed=[f.split('/')[-1].split('.')[0] for f in files]
    completed=[c.split('_')[-2]+','+c.split('_')[-1] for c in completed]
else:
    completed=[]

#all left to do
todo =[]
for t in tiles:  
    if not t in completed: todo.append(t)

#split into jobs
def split(arr, count):
    chunk = int(ceil(len(arr)*1./count))
    return [arr[i::chunk] for i in range(chunk)]

tilelists=split(todo, ntile_per_job)
for jobid, tilelist in enumerate(tilelists):
    with open('{0}_{1:02d}'.format(output_basename, jobid),'w') as output:
        for t in tilelist: print(t, file=output)
        
