#!/bin/bash

##########################
# SMP Script for CANDIDE #
##########################
# Receive email when job finishes or aborts
#PBS -M tobias.liaudat@cea.fr
#PBS -m ea
# Set a name for the job
#PBS -N test_low_den_bigmodel_deep_mccd
# Join output and errors in one file
#PBS -j oe
# Set maximum computing time (e.g. 5min)
#PBS -l walltime=95:00:00
# Request number of cores
#PBS -l nodes=n03:ppn=3:hasgpu

# Full path to environment
export SPENV="$HOME/.conda/envs/shapepipe"
export CONFDIR="$HOME/github/aziz_repos/deep_mccd"

# Activate conda environment
module load tensorflow/2.4
module load intel/19.0/2
source activate shapepipe

# Run ShapePipe using full paths to executables
# Flat SNR
/home/tliaudat/.local/bin/shapepipe_run.py -c $CONFDIR/jobs/testing_jobs/testing_configs/test_pipe_deep_mccd_bigmodel_low_den_L256.ini


# Return exit code
exit 0
