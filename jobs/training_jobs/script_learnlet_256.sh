#!/bin/bash

##########################
# SMP Script for CANDIDE #
##########################
# Receive email when job finishes or aborts
#PBS -M tobias.liaudat@cea.fr
#PBS -m ea
# Set a name for the job
#PBS -N learnlet_train_glob
# Join output and errors in one file
#PBS -j oe
# Set maximum computing time (e.g. 5min)
#PBS -l walltime=90:00:00
# Request number of cores
#PBS -l nodes=n16:ppn=4:hasgpu

# Full path to environment
export CONFDIR="$HOME/github/aziz_repos/deep_mccd"

# Activate conda environment
module purge
module load tensorflow/2.4
# module load intel/19.0/2
source activate shapepipe

# Run ShapePipe using full paths to executables
python $CONFDIR/scripts/training_learnlets.py \
    --run_id_name learnlet_256 \
    --dataset_path /n05data/ayed/outputs/eigenpsfs/dataset_eigenpsfs.fits \
    --base_save_path /n05data/tliaudat/new_deepmccd/reproduce_aziz_results/trained_nets/learnlets_256/ \
    --n_tiling 256 \
    --n_scales 5 \
    --use_lr_scheduler True \
    --enhance_noise True \
    --n_shuffle 20 \
    --batch_size 32 \
    --n_epochs 500 \
    --lr_param 1e-3 \
    --data_train_ratio 0.9 \
    --snr_range 1e-3 50 \

# Return exit code
exit 0
