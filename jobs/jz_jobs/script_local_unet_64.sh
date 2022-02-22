#!/bin/bash
#SBATCH --job-name=local_unet_64    # nom du job
##SBATCH --partition=gpu_p2          # de-commente pour la partition gpu_p2
#SBATCH --ntasks=1                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1                 # nombre de GPU par n?~Sud (max 8 avec gpu_p2)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (un quart du noeud ici)
#SBATCH -C v100-32g 
# /!\ Attention, "multithread" fait reference a l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=100:00:00              # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=local_unet_64_%j.out  # nom du fichier de sortie
#SBATCH --error=local_unet_64_%j.err   # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH -A ynx@gpu                   # specify the project
##SBATCH --qos=qos_gpu-dev            # using the dev queue, as this is only for profiling
#SBATCH --qos=qos_gpu-t4              # We need a long run

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
module load tensorflow-gpu/py3/2.4.1

# echo des commandes lancees
set -x

cd $WORK/repo/deep_mccd/scripts/

srun python -u training_unets.py \
    --run_id_name local_unet_64 \
    --dataset_path /gpfswork/rech/ynx/ulx23va/deep_mccd_project/eigenpsf_datasets/local_eigenpsfs.fits \
    --base_save_path /gpfswork/rech/ynx/ulx23va/deep_mccd_project/trained_models/local_unet_64/ \
    --layers_n_channel 64 \
    --layers_levels 5 \
    --kernel_size 3 \
    --use_lr_scheduler True \
    --enhance_noise True \
    --n_shuffle 50 \
    --batch_size 64 \
    --n_epochs 500 \
    --lr_param 1e-3 \
    --data_train_ratio 0.9 \
    --snr_range 1e-3 100 \

