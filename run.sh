#!/bin/bash

for bs in {1..16};
do
	for i in {1..10};
	do
		qsub -v args="$bs" 2_start.sh
	done
done

for bs in {18..64..2};
do
        for i in {1..10};
        do
                qsub -v args="$bs" 2_start.sh
        done
done

for bs in {68..128..4};
do
        for i in {1..10};
        do
                qsub -v args="$bs" 2_start.sh
        done
done


