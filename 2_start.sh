#!/bin/bash
#PBS -N 3_seO
#PBS -l select=1:ncpus=2:ngpus=1:cluster=glados:mem=24gb:scratch_local=40gb
#PBS -l walltime=23:59:00
#PBS -q gpu
#PBS -m n

INPUT=/storage/brno3-cerit/home/sejakm/mnistest

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR"

#. /storage/brno3-cerit/home/$USER/setup_env.sh

#export PYTHONUSERBASE=/storage/brno3-cerit/home/$USER/.local
#export PATH=$PYTHONUSERBASE/bin:$PATH
#export PYTHONPATH=$PYTHONUSERBASE/lib/python3.8/site-packages:$PYTHONPATH
#export PYTHONPATH=.:$SCRATCHDIR/src:$SCRATCHDIR:$PYTHONPATH

cp -r $INPUT/* $SCRATCHDIR
cd $SCRATCHDIR

echo "WORKING DIRECTORY"
pwd
echo "CONTENTS"
ls -al
echo "EXECUTING..."

module add conda-modules
conda activate sm_custom
pip list > $INPUT/OUTPUT_pip_list.txt

#python $SCRATCHDIR/test.py
python $SCRATCHDIR/main.py $args

clean_scratch
