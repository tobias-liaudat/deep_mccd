#!/bin/bash

##########################
# SMP Script for CANDIDE #
##########################
# Receive email when job finishes or aborts
#PBS -M aziz.ayed@cea.fr
#PBS -m ea
# Set a name for the job
#PBS -N shapepipe_smp
# Join output and errors in one file
#PBS -j oe
# Set maximum computing time (e.g. 5min)
#PBS -l walltime=90:00:00
# Request number of cores
#PBS -l nodes=n16:ppn=2:hasgpu
# Full path to environment
export SPENV="$HOME/.conda/envs/shapepipe"
export CONFDIR="$HOME/github"
# Activate conda environment
module purge
module load tensorflow/2.4
# Run ShapePipe using full paths to executables
## $SPENV/bin/shapepipe_run -c $CONFDIR/mccd_test14.ini
python $CONFDIR/denoising/submission_scripts/training_learnlets/learnlets_256.py
# Return exit code
exit 0
