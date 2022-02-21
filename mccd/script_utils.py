

import sys
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from mccd.denoising.learnlets.learnlet_model import Learnlet
from mccd.denoising.unets.unet import Unet
from mccd.denoising.evaluate import keras_psnr, center_keras_psnr
from mccd.denoising.preprocessing import eigenPSF_data_gen

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def train_learnlets(**args):
    r""" Train the Learnlet model.
    
    """
    # Paths
    run_id_name = args['run_id_name']
    eigenpsf_dataset_path = args['dataset_path']
    base_save_path = args['base_save_path']
    checkpoint_path = base_save_path + 'cp_' + run_id_name + '.h5'

    # Training parameters
    batch_size = args['batch_size'] # 32
    n_epochs = args['n_epochs'] # 500
    lr_param = args['lr_param'] # 1e-3

    # Learnlet parameters
    run_params = {
        'denoising_activation': 'dynamic_soft_thresholding',
        'learnlet_analysis_kwargs':{
            'n_tiling': args['n_tiling'],
            'mixing_details': False,
            'skip_connection': True,
        },
        'learnlet_synthesis_kwargs': {
            'res': True,
        },
        'threshold_kwargs':{
            'noise_std_norm': True,
        },
    #     'wav_type': 'bior',
        'n_scales': args['n_scales'],
        'n_reweights_learn': 1,
        'clip': False,
    }

    # Save output prints to logfile
    old_stdout = sys.stdout
    log_file = open(base_save_path + run_id_name + '_output.log','w')
    sys.stdout = log_file
    print('Starting the log file.')

    print(tf.test.gpu_device_name())

    print('Load data..')
    img = fits.open(eigenpsf_dataset_path)
    img = img[1].data['VIGNETS_NOISELESS']

    np.random.shuffle(img)

    size_train = np.floor(len(img) * args['data_train_ratio'])
    training, test = img[:int(size_train),:,:], img[int(size_train):,:,:]

    print('Prepare datasets..')
    training = eigenPSF_data_gen(
        data=training,
        snr_range=args['snr_range'], # [1e-3, 50],
        img_shape=(51, 51),
        batch_size=batch_size,
        n_shuffle=args['n_shuffle']
    )

    test = eigenPSF_data_gen(
        data=test,
        snr_range=args['snr_range'], #[1e-3, 50],
        img_shape=(51, 51),
        batch_size=1
    )


    model = Learnlet(**run_params)
    steps = int(size_train/batch_size)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='mse',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='min',
        save_freq='epoch',
        options=None
    )

    def l_rate_schedule(epoch):
        return max(1e-3 / 2**(epoch//25), 1e-5)
    lr_cback = tf.keras.callbacks.LearningRateScheduler(l_rate_schedule)

    if args['use_lr_scheduler']:
        models_callbacks = [cp_callback, lr_cback]
    else:
        models_callbacks = [cp_callback]

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_param),
        loss='mse',
        metrics=['mse', keras_psnr, center_keras_psnr],
    )

    print('Start model training and timing..')
    start_train = time.time()
    history = model.fit(
        training,
        validation_data=test,
        steps_per_epoch=steps,
        epochs=n_epochs,
        validation_steps=1,
        callbacks=models_callbacks,
        shuffle=False,
        verbose=2,
    )
    print('Model training ended..')
    end_train = time.time()
    print('Train elapsed time: %f'%(end_train-start_train))

    # Save history file
    try:
        np.save(base_save_path + run_id_name + '_history_file.npy', history.history, allow_pickle=True)
    except:
        pass

    plt.figure(figsize=(12,8))
    plt.plot(history.history['loss'], label='Loss (training data)')
    plt.plot(history.history['val_loss'], label='Loss (validation data)')
    plt.title('Loss of the ' + run_id_name + ' on the EigenPSF Dataset')
    plt.ylabel('Loss value')
    plt.yscale('log')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.savefig(base_save_path + run_id_name +'_history_plot.pdf')

    with open(base_save_path + run_id_name + '_model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    ## Close log file
    print('\n Good bye..')
    sys.stdout = old_stdout
    log_file.close()

def train_unets(**args):
    r""" Train the unets model.
    
    """
    # Paths
    run_id_name = args['run_id_name']
    eigenpsf_dataset_path = args['dataset_path']
    base_save_path = args['base_save_path']
    checkpoint_path = base_save_path + 'cp_' + run_id_name + '.h5'

    # Training parameters
    batch_size = args['batch_size'] # 32
    n_epochs = args['n_epochs'] # 500
    lr_param = args['lr_param'] # 1e-3

    # Unet parameters

    # Save output prints to logfile
    old_stdout = sys.stdout
    log_file = open(base_save_path + run_id_name + '_output.log','w')
    sys.stdout = log_file
    print('Starting the log file.')

    print(tf.test.gpu_device_name())

    print('Load data..')
    img = fits.open(eigenpsf_dataset_path)
    img = img[1].data['VIGNETS_NOISELESS']

    np.random.shuffle(img)

    size_train = np.floor(len(img) * args['data_train_ratio'])
    training, test = img[:int(size_train),:,:], img[int(size_train):,:,:]

    print('Prepare datasets..')
    training = eigenPSF_data_gen(
        data=training,
        snr_range=args['snr_range'], # [1e-3, 50],
        img_shape=(51, 51),
        batch_size=batch_size,
        n_shuffle=args['n_shuffle'],
        noise_estimator=False,
    )

    test = eigenPSF_data_gen(
        data=test,
        snr_range=args['snr_range'], #[1e-3, 50],
        img_shape=(51, 51),
        batch_size=1,
        noise_estimator=False,
    )

    steps = int(size_train/batch_size)

    # Increasing the filter number with a factor of 2
    layers_n_channels = [args['layers_n_channel'] * (2**it) for it in range(args['layers_levels'])]
    print('layers_n_channels: ', layers_n_channels)

    model = Unet(
        n_output_channels=1,
        kernel_size=args['kernel_size'],
        layers_n_channels=layers_n_channels
    )    

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='mse',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='min',
        save_freq='epoch',
        options=None
    )

    def l_rate_schedule(epoch):
        return max(1e-3 / 2**(epoch//25), 1e-5)
    lr_cback = tf.keras.callbacks.LearningRateScheduler(l_rate_schedule)

    if args['use_lr_scheduler']:
        models_callbacks = [cp_callback, lr_cback]
    else:
        models_callbacks = [cp_callback]

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_param),
        loss='mse',
        metrics=['mse', keras_psnr, center_keras_psnr],
    )

    print('Start model training and timing..')
    start_train = time.time()
    history = model.fit(
        training,
        validation_data=test,
        steps_per_epoch=steps,
        epochs=n_epochs,
        validation_steps=1,
        callbacks=models_callbacks,
        shuffle=False,
        verbose=2,
    )
    print('Model training ended..')
    end_train = time.time()
    print('Train elapsed time: %f'%(end_train-start_train))

    # Save history file
    try:
        np.save(base_save_path + run_id_name + '_history_file.npy', history.history, allow_pickle=True)
    except:
        pass

    plt.figure(figsize=(12,8))
    plt.plot(history.history['loss'], label='Loss (training data)')
    plt.plot(history.history['val_loss'], label='Loss (validation data)')
    plt.title('Loss of the ' + run_id_name + ' on the EigenPSF Dataset')
    plt.ylabel('Loss value')
    plt.yscale('log')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.savefig(base_save_path + run_id_name +'_history_plot.pdf')

    with open(base_save_path + run_id_name + '_model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    ## Close log file
    print('\n Good bye..')
    sys.stdout = old_stdout
    log_file.close()
