#!/bin/bash

# iterate over tilenames

lines=`find batch_jobs -maxdepth 1 -name "jobs_tilelist_??" |sort ` #-printf "%f\n"`
for line in $lines; do
    echo $line
    #qsub -v TILELIST=$line jobs_2016_2017.pbs
done
wait
