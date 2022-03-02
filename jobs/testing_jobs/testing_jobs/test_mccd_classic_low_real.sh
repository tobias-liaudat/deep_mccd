#!/bin/bash

##########################
# SMP Script for CANDIDE #
##########################
# Receive email when job finishes or aborts
#PBS -M tobias.liaudat@cea.fr
#PBS -m ea
# Set a name for the job
#PBS -N test_classic_mccd
# Join output and errors in one file
#PBS -j oe
# Set maximum computing time (e.g. 5min)
#PBS -l walltime=25:00:00
# Request number of cores
#PBS -l nodes=n02:ppn=3

# Full path to environment
export SPENV="$HOME/.conda/envs/shapepipe_mccd"
export CONFDIR="$HOME/github/aziz_repos/deep_mccd"

# Activate conda environment
module purge
module load intelpython/3-2020.1
module load intel/19.0/2
source activate shapepipe_mccd

# Run ShapePipe using full paths to executables
# /home/tliaudat/.local/bin/shapepipe_run.py -c $CONFDIR/config_files/mccd_eigen_generation.ini
$SPENV/bin/shapepipe_run -c $CONFDIR/jobs/testing_jobs/testing_configs/test_pipe_classic_mccd_real_wav3.ini

# Return exit code
exit 0
