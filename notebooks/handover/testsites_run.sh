#specify the number of tiles to run
for node in $(seq 0 3);do
    echo $node
    qsub -v si=$node  jobs.pbs #submit individual jobs for each tile

done
wait
