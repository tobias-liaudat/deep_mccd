
# Location of useful files

### Previous version

Aziz trained models (Learnlets)
``` candide
/n05data/ayed/outputs/mccd_runs/high_density/ -> fitted_model-1234567.npy
/n05data/ayed/outputs/mccd_runs/mid_density/ -> fitted_model-1111111.npy
/n05data/ayed/outputs/mccd_runs/low_density/ -> fitted_model-1000000.npy
```

Extracted eigenPSFs from the trained MCCDs. Used for the training of Learn lets and Unets
``` candide
/n05data/ayed/outputs/eigenpsfs/
-> dataset_eigenpsfs.fits
-> global_eigenpsfs.fits
-> local_eigenpsfs.fits
```


Aziz datasets for testing PSF models with different star densities
``` candide
/n05data/ayed/outputs/datasets/final_test/
-> high_density
-> mid_density
-> low_density
```



***

## New version

### Realistic dataset for training and extracting noiseless eigenPSFs
- Total number of generated datasets: 250
``` candide
/n05data/tliaudat/new_deepmccd/training_realistic_sims/inputs/
-> train_star_selection-22*.fits
-> test_star_selection-22*.fits
```

### Trained classical MCCD models used for extracting the eigenPSFs
- Total number of models trained and validated: 500 (7.6G trained models and 82G validation files)
- First dataset starting with 2200..., second dataset starting with 2300...
``` candide
/n05data/tliaudat/new_deepmccd/training_realistic_sims/output_mccd/validation_files/validation_psf-*.fits
/n05data/tliaudat/new_deepmccd/training_realistic_sims/output_mccd/trained_models/fitted_model-*.npy
```

### Baseline performance of classic MCCD without noise
- Performance plots computed with the previous 230 validated datasets.
``` candide
/n05data/tliaudat/new_deepmccd/training_realistic_sims/output_mccd/merge_plot/
```

### EigenPSF datasets
``` candide
/n05data/tliaudat/new_deepmccd/training_realistic_sims/output_mccd/eigenPSF_datasets/
```

