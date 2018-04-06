#!/bin/bash


for YEAR in $(seq 2014 2017)
do
	for mi in $(seq 1 12)
	do
		qsub -v Year=$YEAR,mon=$mi Monthly_GeometricMedian.pbs
	done
done
wait
