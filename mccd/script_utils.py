

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

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


def init_learnlets(weights_path, **args):
    """ Initialise the Learnlet model.

    """
    # Learnlet parameters
    run_params = {
        'denoising_activation': 'dynamic_soft_thresholding',
        'n_scales': args['n_scales'],
        'n_reweights_learn': 1,
        'clip': False,
        'normalize': True,
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
    }

    # Create fake data to init the model
    im_val = tf.convert_to_tensor(np.random.rand(2, args['im_shape'][0], args['im_shape'][1], 1))
    std_val = tf.convert_to_tensor(np.random.rand(2))

    learnlet_model = Learnlet(**run_params)
    # Compile 
    learnlet_model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-6),
        loss='mse',
    )
    # Init
    learnlet_model.evaluate(
        (im_val, std_val),
        im_val,
    )
    # Load and good bye
    learnlet_model.load_weights(weights_path)

    return learnlet_model


def init_unets(weights_path, **args):
    """ Initialise the Unet model.

    """

    # Create fake data to init the model
    im_val = tf.convert_to_tensor(np.random.rand(2, args['im_shape'][0], args['im_shape'][1], 1))

    # Increasing the filter number with a factor of 2
    layers_n_channels = [args['layers_n_channel'] * (2**it) for it in range(args['layers_levels'])]

    # Create instance
    unet_model = Unet(
        n_output_channels=1,
        kernel_size=args['kernel_size'],
        layers_n_channels=layers_n_channels
    )    
    # Compile
    unet_model.compile(
        optimizer = tf.keras.optimizers.Adam(lr=1e-6),
        loss='mse',
    )
    # Init
    unet_model.evaluate(im_val, im_val)
    # Load weights and good bye
    unet_model.load_weights(weights_path)

    return unet_model




def train_learnlets(**args):
    r""" Train the Learnlet model.
    
    """
    # Paths
    run_id_name = args['run_id_name']
    eigenpsf_dataset_path = args['dataset_path']
    base_save_path = args['base_save_path']
    checkpoint_path = base_save_path + 'cp_' + run_id_name + '.h5'

    # Save parameters
    np.save(base_save_path + 'params_' + run_id_name + '.npy', args, allow_pickle=True)

    # Training parameters
    batch_size = args['batch_size'] # 32
    n_epochs = args['n_epochs'] # 500
    lr_param = args['lr_param'] # 1e-3

    # Learnlet parameters
    run_params = {
        'denoising_activation': 'dynamic_soft_thresholding',
        'n_scales': args['n_scales'],
        'n_reweights_learn': 1,
        'clip': False,
        'normalize': True,
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
        n_shuffle=args['n_shuffle'],
        enhance_noise=args['enhance_noise'],
        add_parametric_data=args['add_parametric_data'],
        star_ratio=args['star_ratio'],
        e1_range=args['e1_range'], 
        e2_range=args['e2_range'],
        fwhm_range=args['fwhm_range'],
        pix_scale=args['pix_scale'],
        beta_psf=args['beta_psf'],
    )

    test = eigenPSF_data_gen(
        data=test,
        snr_range=args['snr_range'], #[1e-3, 50],
        img_shape=(51, 51),
        batch_size=1,
        enhance_noise=args['enhance_noise'],
        add_parametric_data=args['add_parametric_data'],
        star_ratio=args['star_ratio'],
        e1_range=args['e1_range'], 
        e2_range=args['e2_range'],
        fwhm_range=args['fwhm_range'],
        pix_scale=args['pix_scale'],
        beta_psf=args['beta_psf'],
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

    # Save parameters
    np.save(base_save_path + 'params_' + run_id_name + '.npy', args, allow_pickle=True)

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
        enhance_noise=args['enhance_noise'],
        add_parametric_data=args['add_parametric_data'],
        star_ratio=args['star_ratio'],
        e1_range=args['e1_range'], 
        e2_range=args['e2_range'],
        fwhm_range=args['fwhm_range'],
        pix_scale=args['pix_scale'],
        beta_psf=args['beta_psf'],
    )

    test = eigenPSF_data_gen(
        data=test,
        snr_range=args['snr_range'], #[1e-3, 50],
        img_shape=(51, 51),
        batch_size=1,
        noise_estimator=False,
        enhance_noise=args['enhance_noise'],
        add_parametric_data=args['add_parametric_data'],
        star_ratio=args['star_ratio'],
        e1_range=args['e1_range'], 
        e2_range=args['e2_range'],
        fwhm_range=args['fwhm_range'],
        pix_scale=args['pix_scale'],
        beta_psf=args['beta_psf'],
    )

    steps = int(size_train/batch_size)

    # Increasing the filter number with a factor of 2
    layers_n_channels = [
        args['layers_n_channel'] * (2**it) 
        for it in range(args['layers_levels'])
    ]
    print('layers_n_channels: ', layers_n_channels)

    model = Unet(
        n_output_channels=1,
        kernel_size=args['kernel_size'],
        layers_n_channels=layers_n_channels,
        spectral_normalization=args['spectral_normalization'],
        power_iterations=args['power_iterations'],
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
