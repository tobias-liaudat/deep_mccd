#!/bin/bash

##########################
# SMP Script for CANDIDE #
##########################
# Receive email when job finishes or aborts
#PBS -M tobias.liaudat@cea.fr
#PBS -m ea
# Set a name for the job
#PBS -N eigen_generation
# Join output and errors in one file
#PBS -j oe
# Set maximum computing time (e.g. 5min)
#PBS -l walltime=99:00:00
# Request number of cores
#PBS -l nodes=n02:ppn=20

# Full path to environment
export SPENV="$HOME/.conda/envs/shapepipe_mccd"
export CONFDIR="$HOME/github/aziz_repos/deep_mccd/jobs"

# Activate conda environment
module load intelpython/3-2020.1
module load intel/19.0/2
source activate shapepipe_mccd

# Run ShapePipe using full paths to executables
# /home/tliaudat/.local/bin/shapepipe_run.py -c $CONFDIR/config_files/mccd_eigen_generation.ini
$SPENV/bin/shapepipe_run -c $CONFDIR/config_files/mccd_eigen_generation.ini

# Return exit code
exit 0
