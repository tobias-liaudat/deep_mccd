#!/usr/bin/env python
# coding: utf-8

import numpy as np
import mccd.dataset_generation as dataset_generation
from joblib import Parallel, delayed, cpu_count, parallel_backend

# Total number of exposure per datasets
n_procs = 3
n_cpus = 3


# Print some info
cpu_info = ' - Number of available CPUs: {}'.format(cpu_count())
proc_info = ' - Total number of processes: {}'.format(n_procs)

# Paths
base_path = '/home/tliaudat/github/aziz_repos/deep_mccd/data/realistic_dataset_input'
e1_path = base_path + '/e1_psf.npy'
e2_path = base_path + '/e2_psf.npy'
fwhm_path = base_path + '/seeing_distribution.npy'
SNR_dist_path = base_path + '/SNR_dist.npy'
base_output_path = '/n05data/tliaudat/new_deepmccd/testing_realistic_sims/inputs/real_SNR/'

# Parameters
SNR_range = None
use_SNR_dist = True
x_grid = 10
y_grid = 20

# Generate catalog list
low_cat_id_list = [3400000 + i for i in range(n_procs)]
mid_cat_id_list = [3500000 + i for i in range(n_procs)]
high_cat_id_list = [3600000 + i for i in range(n_procs)]

# Print info
print('Dataset generation.')
print(cpu_info)
print(proc_info)
print('Number of catalogs: ', n_procs)
print('Number of CPUs: ', n_cpus)

def generate_low_density_dataset(cat_id):
    print('\nProcessing catalog: ', cat_id)
    sim_dataset_generator = dataset_generation.GenerateRealisticDataset(
        e1_path=e1_path,
        e2_path=e2_path,
        size_path=fwhm_path,
        range_mean_star_qt=[15, 15],
        range_dev_star_nb=[-1, 1], 
        save_realisation=False,
        output_path=base_output_path + 'low_density/',
        SNR_dist_path=SNR_dist_path,
        catalog_id=cat_id)
    sim_dataset_generator.generate_train_data(
        use_SNR_dist=use_SNR_dist,
        SNR_range=SNR_range,
    )
    sim_dataset_generator.generate_test_data(
        grid_pos_bool=True,
        x_grid=x_grid,
        y_grid=y_grid,
        SNR_range=None,
        use_SNR_dist=False,
    )

def generate_mid_density_dataset(cat_id):
    print('\nProcessing catalog: ', cat_id)
    sim_dataset_generator = dataset_generation.GenerateRealisticDataset(
        e1_path=e1_path,
        e2_path=e2_path,
        size_path=fwhm_path,
        range_mean_star_qt=[25, 25],
        range_dev_star_nb=[-1, 1], 
        save_realisation=False,
        output_path=base_output_path + 'mid_density/',
        SNR_dist_path=SNR_dist_path,
        catalog_id=cat_id)
    sim_dataset_generator.generate_train_data(
        use_SNR_dist=use_SNR_dist,
        SNR_range=SNR_range,
    )
    sim_dataset_generator.generate_test_data(
        grid_pos_bool=True,
        x_grid=x_grid,
        y_grid=y_grid,
        SNR_range=None,
        use_SNR_dist=False,
    )

def generate_high_density_dataset(cat_id):
    print('\nProcessing catalog: ', cat_id)
    sim_dataset_generator = dataset_generation.GenerateRealisticDataset(
        e1_path=e1_path,
        e2_path=e2_path,
        size_path=fwhm_path,
        range_mean_star_qt=[50, 50],
        range_dev_star_nb=[-1, 1], 
        save_realisation=False,
        output_path=base_output_path + 'high_density/',
        SNR_dist_path=SNR_dist_path,
        catalog_id=cat_id)
    sim_dataset_generator.generate_train_data(
        use_SNR_dist=use_SNR_dist,
        SNR_range=SNR_range,
    )
    sim_dataset_generator.generate_test_data(
        grid_pos_bool=True,
        x_grid=x_grid,
        y_grid=y_grid,
        SNR_range=None,
        use_SNR_dist=False,
    )

# Generate low denisty
with parallel_backend("loky", inner_max_num_threads=1):
    results = Parallel(n_jobs=n_cpus)(delayed(generate_low_density_dataset)(_cat_id)
                                        for _cat_id in low_cat_id_list)

# Generate mid denisty
with parallel_backend("loky", inner_max_num_threads=1):
    results = Parallel(n_jobs=n_cpus)(delayed(generate_mid_density_dataset)(_cat_id)
                                        for _cat_id in mid_cat_id_list)

# Generate high denisty
with parallel_backend("loky", inner_max_num_threads=1):
    results = Parallel(n_jobs=n_cpus)(delayed(generate_high_density_dataset)(_cat_id)
                                        for _cat_id in high_cat_id_list)

