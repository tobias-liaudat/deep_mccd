#!/usr/bin/env python
# coding: utf-8

import numpy as np
import glob
import mccd

# Paths
input_dir_path = '/n05data/tliaudat/new_deepmccd/training_realistic_sims/output_mccd/trained_models/'
model_pattern = 'fitted_model-*.npy'
save_dir_path = '/n05data/tliaudat/new_deepmccd/training_realistic_sims/output_mccd/eigenPSF_datasets/'

# Check the files that match the pattern in the input dir
file_paths = glob.glob(input_dir_path + model_pattern)


## Extracting the EigenPSFs from the fitted models
global_eigenPSFs = []
local_eigenPSFs = []

for path in file_paths:
    
    fitted_model = np.load(path, allow_pickle=True)
    S = fitted_model[1]['S']
    
    global_eigenPSFs.append(mccd.utils.reg_format(S[-1]))

    for k in range (40):
        local_eigenPSFs.append(mccd.utils.reg_format(S[k]))

# Concatenate list to np.ndarray
global_eigenPSFs = np.concatenate(global_eigenPSFs, axis=0)
local_eigenPSFs = np.concatenate(local_eigenPSFs, axis=0)

# Shuffle vignets
np.random.shuffle(global_eigenPSFs)
np.random.shuffle(local_eigenPSFs)


# Save files
save_dic = {'VIGNETS_NOISELESS': global_eigenPSFs}
mccd.mccd_utils.save_to_fits(save_dic, save_dir_path + 'global_eigenpsfs.fits')

save_dic = {'VIGNETS_NOISELESS': local_eigenPSFs}
mccd.mccd_utils.save_to_fits(save_dic, save_dir_path + 'local_eigenpsfs.fits')

save_dic = {'VIGNETS_NOISELESS': np.concatenate((global_eigenPSFs, local_eigenPSFs), axis=0)}
mccd.mccd_utils.save_to_fits(save_dic, save_dir_path + 'all_eigenpsfs.fits')

    