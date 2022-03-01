#!/bin/bash
#SBATCH --job-name=train_U32    # nom du job
##SBATCH --partition=gpu_p2          # de-commente pour la partition gpu_p2
#SBATCH --ntasks=1                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1                 # nombre de GPU par n?~Sud (max 8 avec gpu_p2)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (un quart du noeud ici)
#SBATCH -C v100-32g 
# /!\ Attention, "multithread" fait reference a l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=100:00:00              # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=train_U32_%j.out  # nom du fichier de sortie
#SBATCH --error=train_U32_%j.err   # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH -A ynx@gpu                   # specify the project
##SBATCH --qos=qos_gpu-dev            # using the dev queue, as this is only for profiling
#SBATCH --qos=qos_gpu-t4              # We need a long run
#SBATCH --array=0-3

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
module load tensorflow-gpu/py3/2.4.1

# echo des commandes lancees
set -x

cd $WORK/repo/deep_mccd/scripts/

opt[0]="--run_id_name global_enhanced_U32 --n_epochs 500 --enhance_noise True --dataset_path /gpfswork/rech/ynx/ulx23va/deep_mccd_project/eigenpsf_datasets/global_eigenpsfs.fits"
opt[1]="--run_id_name global_flat_U32 --n_epochs 500 --enhance_noise False --dataset_path /gpfswork/rech/ynx/ulx23va/deep_mccd_project/eigenpsf_datasets/global_eigenpsfs.fits"
opt[2]="--run_id_name local_enhanced_U32 --n_epochs 150 --enhance_noise True --dataset_path /gpfswork/rech/ynx/ulx23va/deep_mccd_project/eigenpsf_datasets/filtered_local_eigenpsfs.fits"
opt[3]="--run_id_name local_flat_U32 --n_epochs 150 --enhance_noise False --dataset_path /gpfswork/rech/ynx/ulx23va/deep_mccd_project/eigenpsf_datasets/filtered_local_eigenpsfs.fits"


srun python -u training_unets.py \
    --dataset_path /gpfswork/rech/ynx/ulx23va/deep_mccd_project/eigenpsf_datasets/global_eigenpsfs.fits \
    --base_save_path /gpfswork/rech/ynx/ulx23va/deep_mccd_project/trained_models/testing_models/U32/ \
    --layers_n_channel 32 \
    --layers_levels 5 \
    --kernel_size 3 \
    --spectral_normalization True \
    --power_iterations 1 \
    --use_lr_scheduler True \
    --n_shuffle 50 \
    --batch_size 32 \
    --lr_param 1e-3 \
    --data_train_ratio 0.9 \
    --snr_range 1e-3 100 \
    --add_parametric_data True \
    --star_ratio 0.4 \
    ${opt[$SLURM_ARRAY_TASK_ID]} \
