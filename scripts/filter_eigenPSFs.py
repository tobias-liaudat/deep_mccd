#!/usr/bin/env python
# coding: utf-8

import numpy as np
import mccd
from astropy.io import fits

# Load the data
local_dataset_path = '/n05data/tliaudat/new_deepmccd/training_realistic_sims/output_mccd/eigenPSF_datasets/local_eigenpsfs.fits'
global_dataset_path = '/n05data/tliaudat/new_deepmccd/training_realistic_sims/output_mccd/eigenPSF_datasets/global_eigenpsfs.fits'

save_dir_path = '/n05data/tliaudat/new_deepmccd/training_realistic_sims/output_mccd/eigenPSF_datasets/'

dataset = fits.open(local_dataset_path)
print(dataset[1].data['VIGNETS_NOISELESS'].shape)


# Design the filter
butter_filt_5_20 = mccd.utils.butterworth_2d_filter(im_shape=(51,51), order=5, cut_dist=20)
# Filter
filtered_eigenPSF = np.array([
    mccd.utils.fft_filter_image(eig, butter_filt_5_20)
    for eig in dataset[1].data['VIGNETS_NOISELESS']
])

# Save new filtered local eigenPSFs
save_dic = {'VIGNETS_NOISELESS': filtered_eigenPSF}
mccd.mccd_utils.save_to_fits(save_dic, save_dir_path + 'filtered_local_eigenpsfs.fits')

# Load global dataset
global_dataset = fits.open(global_dataset_path)
print(global_dataset[1].data['VIGNETS_NOISELESS'].shape)

# Concatenate and shuffle
all_eigenPSFs = np.concatenate((filtered_eigenPSF, global_dataset[1].data['VIGNETS_NOISELESS']), axis=0)
np.random.shuffle(all_eigenPSFs)

# Save the new dataset with all the eigenPSFs
save_dic = {'VIGNETS_NOISELESS': all_eigenPSFs}
mccd.mccd_utils.save_to_fits(save_dic, save_dir_path + 'all_filtered_local_eigenpsfs.fits')
