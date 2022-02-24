
cd /Users/tliaudat/Documents/PhD/github/cosmostat_official/deep_mccd/deep_mccd/scripts/



/Users/tliaudat/opt/anaconda3/envs/current_shapepipe/bin/python training_learnlets.py \
    --run_id_name local_learnlet_512 \
    --dataset_path /gpfswork/rech/ynx/ulx23va/deep_mccd_project/eigenpsf_datasets/local_eigenpsfs.fits \
    --base_save_path /gpfswork/rech/ynx/ulx23va/deep_mccd_project/trained_models/local_learnlet_512/ \
    --n_tiling 512 \
    --n_scales 5 \
    --use_lr_scheduler True \
    --enhance_noise True \
    --n_shuffle 50 \
    --batch_size 64 \
    --n_epochs 500 \
    --lr_param 1e-3 \
    --data_train_ratio 0.9 \
    --snr_range 1e-3 100 \
