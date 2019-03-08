

#tiles=(199 375 424 537 576 815 816 817 818 838 839 841 863 864 865 866 889)

#for ti in $(seq 640  928) #
for ti in "${tiles[@]}"
do
echo $ti 
year=2016
method='NBRdist'
jobidx=` qsub -v ti=$ti,year=$year,method=$method jobs.pbs`
jobidx2=` qsub -W depend=afterok:${jobidx} -v ti=$ti,year=$year,method=$method merge_tiles.pbs`
echo $jobidx $jobidx2
sleep 1

done
wait

