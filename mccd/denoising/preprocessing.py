#!/usr/bin/env python
# coding: utf-8

r"""PREPROCESSING.
Defines the preprocessing functions that will be used to train the model.
"""


import numpy as np
import tensorflow as tf
from astropy.io import fits
import galsim as gs
import itertools

def mad(x):
    r"""Compute an estimation of the standard deviation 
    of a Gaussian distribution using the robust 
    MAD (Median Absolute Deviation) estimator."""
    return 1.4826*np.median(np.abs(x - np.median(x)))

def STD_estimator(image, window):    
    # Calculate noise std dev
    return mad(image[window])


def calculate_window(im_shape=(51, 51), win_rad=14):
    # Calculate window function for estimating the noise
    # We take the center of the image and a large radius to cut all the flux from the star
    window = np.ones(im_shape, dtype=bool)
    
    for coord_x in range(im_shape[0]):
        for coord_y in range(im_shape[1]):
            if np.sqrt((coord_x - im_shape[0]/2)**2 + (coord_y - im_shape[1]/2)**2) <= win_rad :
                window[coord_x, coord_y] = False

    return window

def add_noise_function(image, snr_range, tf_window, noise_estimator=True, enhance_noise=False):
    # Draw random SNR
    snr = tf.random.uniform(
            (1,),
            minval=snr_range[0],
            maxval=snr_range[1],
            dtype=tf.float64
        )

    if enhance_noise:
        snr = tf.random.uniform(
            (1,),
            minval=snr_range[0],
            maxval=snr[0],
            dtype=tf.float64
        )
  
    # Apply the noise
    im_shape = tf.cast(tf.shape(image), dtype=tf.float64)
    sigma_noise = tf.norm(image, ord='euclidean') / (snr * im_shape[0] * im_shape[1])
    noisy_image = image + tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=sigma_noise, dtype=tf.float64)
    # norm_noisy_img = noisy_image / tf.norm(noisy_image, ord='euclidean')

    # Apply window to the normalised noisy image
    windowed_img = tf.boolean_mask(noisy_image, tf_window)
    if noise_estimator:       
        return noisy_image, tf.reshape(tf.numpy_function(mad,[windowed_img], Tout=tf.float64), [1,])
    else:
        return noisy_image

def normalise(image):
    """ Normalise (scale the image values). 
    
    I in [a,b]
    Operations are:
    [0,b-a]
    [0, 1]
    tilde(I) in [-0.5, 0.5]
    """
    image -= tf.math.reduce_min(image)
    image /= tf.math.reduce_max(image)
    return image - 0.5


def eigenPSF_data_gen(
    data, 
    snr_range=[1e-3, 50],
    img_shape=(51, 51),
    batch_size=16,
    win_rad=14,
    n_shuffle=50,
    noise_estimator=True,
    enhance_noise=False,
    add_parametric_data=False,
    star_ratio=0.5,
    e1_range=[-0.15, 0.15],
    e2_range=[-0.15, 0.15],
    fwhm_range=[0.5, 1.],
    pix_scale = 0.187,
    beta_psf = 4.765,
):
    """ Dataset generator of eigen-PSFs.

    On-the-fly addition of noise following a SNR distribution.
    We also calculate the noise std.

    Parameters
    ----------
    data: np.ndarray [batch, im_x, im_y]
        Input images.
    snr_range: numpy.ndarray
        Min and max snr for noise addition.
    img_shape: tuple
        Image dimensions
    batch_size: int
        Batch size
    win_rad: int
        Window radius in pixels for noise estimation.
    n_shuffle: int
        Number of batchs used to shuffle.
    noise_estimator: bool
        If the noise estimation is returned from the `add_noise_function`
    enhance_noise: bool
        If True the noise distribution will be skewed for more noisy values
        insterad of being flat in the SNR range.
    add_parametric_data: bool
        Option to add parametric dataset.
    star_ratio: float
        Ratio of parametric samples on final samples.
    e1_range: (list of) float
        Range of e1 variation.
    e2_range: (list of) float
        Range of e2 variation.
    fwhm_range: (list of) float
        Range of fwhm variation in arcsec.
    pix_scale: float
        Pixel scale for the parametric star simulation.
    beta_psf: float
        Beta parameter of the Moffat profile.
        
    """
    # Verify the eigenPSFs are positive
    multiple = np.array([np.sum(im)>0 for im in data]) * 2. - 1.
    data *= multiple.reshape((-1, 1, 1))
    # Verify consistency
    if (data.shape[1] != img_shape[0]) or (data.shape[2] != img_shape[1]):
        raise ValueError
        
    # Create parametric dataset and concatenate with current dataset
    if add_parametric_data:
        eigen_num = data.shape[0]
        _star_num = np.ceil(eigen_num * star_ratio / (1 - star_ratio)).astype(int)
        var_num = np.ceil(np.power(_star_num, 1/3)).astype(int)
        star_num = var_num**3
        print('Total dataset: ',star_num+eigen_num, ', with eigenPSF ', eigen_num, ', and parametric stars ', star_num)
        # Define the variations
        e1 = np.linspace(e1_range[0], e1_range[1], num=var_num, endpoint=True)
        e2 = np.linspace(e2_range[0], e2_range[1], num=var_num, endpoint=True)
        fwhm = np.linspace(fwhm_range[0], fwhm_range[1],  num=var_num, endpoint=True)
        # Build the datasets arrays
        es = list(itertools.product(*[e1, e2, fwhm]))
        e1s = np.array([a for a,b,c in es])
        e2s = np.array([b for a,b,c in es]) 
        fwhms = np.array([c for a,b,c in es])
        # Fix PSF flux
        psf_flux = 1.
        # Create the array
        parametric_data = np.zeros((star_num, img_shape[0], img_shape[1]))
        
        for it in range(star_num):
            # PSF generation. Define size
            psf = gs.Moffat(fwhm=fwhms[it], beta=beta_psf)
            # Define the Flux
            psf = psf.withFlux(psf_flux)
            # Define the shear
            psf = psf.shear(g1=e1s[it], g2=e2s[it])
            # Draw the PSF on a vignet
            image_epsf = gs.ImageF(img_shape[0], img_shape[1])
            # Define intrapixel shift (uniform distribution in [-0.25,0.25])
            rand_shift = (np.random.rand(2) - 0.5) / 2
            psf.drawImage(image=image_epsf, offset=rand_shift, scale=pix_scale)
            # Save images before adding the noise
            parametric_data[it,:,:] = image_epsf.array
            
        # Concatenate the simulations to the input data
        data = np.concatenate((data, parametric_data), axis=0)
            
    print('data.shape :', data.shape)
    # Expand last dimension
    data = np.reshape(data, (data.shape[0], img_shape[0], img_shape[1], 1))
    # Shuffle data
    np.random.shuffle(data)
    # Init dataset from file
    ds = tf.data.Dataset.from_tensor_slices(data)
    # Cast SNR range 
    tf_snr_range = tf.cast(snr_range, dtype=tf.float64)
    # Create window for noise estimation
    tf_window = tf.cast(calculate_window(
        im_shape=(img_shape[0], img_shape[1]), win_rad=win_rad),
        dtype=tf.bool)
    tf_window = tf.reshape(tf_window, (img_shape[0], img_shape[1], 1))

    # Normalise
    image_noise_ds = ds.map(normalise, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Apply noise and estimate noise std
    image_noisy_ds = image_noise_ds.map(
        lambda x: (add_noise_function(
            x,
            tf_snr_range,
            tf_window, 
            noise_estimator=noise_estimator,
            enhance_noise=enhance_noise
        ), x),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    # Shuffle the data
    image_noise_ds = image_noisy_ds.shuffle(buffer_size=n_shuffle*batch_size)
    # Batch after shuffling to get unique batches at each epoch.
    image_noisy_ds = image_noisy_ds.batch(batch_size)
    image_noisy_ds = image_noisy_ds.repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return image_noisy_ds






