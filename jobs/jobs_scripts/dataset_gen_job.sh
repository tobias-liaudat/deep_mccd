#!/bin/bash

##########################
# SMP Script for CANDIDE #
##########################
# Receive email when job finishes or aborts
#PBS -M tobias.liaudat@cea.fr
#PBS -m ea
# Set a name for the job
#PBS -N dataset_gen
# Join output and errors in one file
#PBS -j oe
# Set maximum computing time (e.g. 5min)
#PBS -l walltime=90:00:00
# Request number of cores
#PBS -l nodes=n02:ppn=12

# Full path to environment
export SPENV="$HOME/.conda/envs/shapepipe_mccd"
export CONFDIR="$HOME/github/aziz_repos/deep_mccd"

# Activate conda environment
module purge
# module load tensorflow/2.4
module load intelpython/3-2020.1
module load intel/19.0/2
source activate shapepipe_mccd

# Run ShapePipe using full paths to executables
python $CONFDIR/scripts/realistic_dataset_generation_parallel.py

# Return exit code
exit 0
