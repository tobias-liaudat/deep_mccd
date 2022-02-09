#!/bin/bash

##########################
# SMP Script for CANDIDE #
##########################
# Receive email when job finishes or aborts
#PBS -M tobias.liaudat@cea.fr
#PBS -m ea
# Set a name for the job
#PBS -N shapepipe_job
# Join output and errors in one file
#PBS -j oe
# Set maximum computing time (e.g. 5min)
#PBS -l walltime=90:00:00
# Request number of cores
#PBS -l nodes=n16:ppn=2:hasgpu

# Full path to environment
export SPENV="$HOME/.conda/envs/shapepipe"
export CONFDIR="$HOME/github/aziz_repos/deep_mccd/jobs"

# Activate conda environment
module load tensorflow/2.4
module load intel/19.0/2
source activate shapepipe

# Run ShapePipe using full paths to executables
$SPENV/bin/shapepipe_run -c $CONFDIR/config_files/deep_mccd.ini

# Return exit code
exit 0


