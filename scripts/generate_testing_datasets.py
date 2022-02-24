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


base_path = '/home/tliaudat/github/aziz_repos/deep_mccd/data/realistic_dataset_input'
e1_path = base_path + '/e1_psf.npy'
e2_path = base_path + '/e2_psf.npy'
fwhm_path = base_path + '/seeing_distribution.npy'
base_output_path = '/n05data/tliaudat/new_deepmccd/testing_realistic_sims/inputs/'

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
        catalog_id=cat_id)
    sim_dataset_generator.generate_train_data()
    sim_dataset_generator.generate_test_data()

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
        catalog_id=cat_id)
    sim_dataset_generator.generate_train_data()
    sim_dataset_generator.generate_test_data()

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
        catalog_id=cat_id)
    sim_dataset_generator.generate_train_data()
    sim_dataset_generator.generate_test_data()


# Generate catalog list
low_cat_id_list = [2500000 + i for i in range(n_procs)]
mid_cat_id_list = [2600000 + i for i in range(n_procs)]
high_cat_id_list = [2700000 + i for i in range(n_procs)]

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

