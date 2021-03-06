#!/usr/bin/env python
# coding: utf-8

import numpy as np
import mccd.dataset_generation as dataset_generation
from joblib import Parallel, delayed, cpu_count, parallel_backend


# Total number of datasets
n_procs = 270
n_cpus = 8


# Print some info
cpu_info = ' - Number of available CPUs: {}'.format(cpu_count())
proc_info = ' - Total number of processes: {}'.format(n_procs)

# Paths
# e1_path = '/n05data/ayed/data/moments/e1_psf.npy'
# e2_path = '/n05data/ayed/data/moments/e2_psf.npy'
# fwhm_path = '/n05data/ayed/data/moments/seeing_distribution.npy'
# output_path = '/n05data/ayed/outputs/datasets/'

base_path = '/home/tliaudat/github/aziz_repos/deep_mccd/data/realistic_dataset_input'
e1_path = base_path + '/e1_psf.npy'
e2_path = base_path + '/e2_psf.npy'
fwhm_path = base_path + '/seeing_distribution.npy'
output_path = '/n05data/tliaudat/new_deepmccd/training_realistic_sims/inputs/'

# Print info
print('Dataset generation.')
print(cpu_info)
print(proc_info)
print('Number of catalogs: ', n_procs)
print('Number of CPUs: ', n_cpus)

# Generate catalog list
cat_id_list = [2300000 + i for i in range(n_procs)]


def generate_dataset(cat_id):
    print('\nProcessing catalog: ', cat_id)
    sim_dataset_generator = dataset_generation.GenerateRealisticDataset(
        e1_path=e1_path,
        e2_path=e2_path,
        size_path=fwhm_path,
        output_path=output_path,
        catalog_id=cat_id)
    sim_dataset_generator.generate_train_data()
    sim_dataset_generator.generate_test_data()


with parallel_backend("loky", inner_max_num_threads=1):
    results = Parallel(n_jobs=n_cpus)(delayed(generate_dataset)(_cat_id)
                                        for _cat_id in cat_id_list)
