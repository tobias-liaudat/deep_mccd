#!/usr/bin/env python
# coding: utf-8

import mccd
import click

@click.command()

## Training options
# Input parameters
@click.option(
    "--run_id_name",
    default="unet_32",
    type=str,
    help="Model id saving name.")
@click.option(
    "--dataset_path",
    default="/n05data/ayed/outputs/eigenpsfs/dataset_eigenpsfs.fits",
    type=str,
    help="Input dataset path.")
# Saving paths
@click.option(
    "--base_save_path",
    default="/n05data/tliaudat/new_deepmccd/reproduce_aziz_results/trained_nets/unet_32/",
    type=str,
    help="Base path for saving files.")
# Learnlet parameters
@click.option(
    "--layers_n_channel",
    default=32,
    type=int,
    help="Number filters in the smallest level of the unet. Increasing a factor of 2 each time we go to the next deeper level.")
@click.option(
    "--layers_levels",
    default=5,
    type=int,
    help="Number deepness levels in the unet.")    
@click.option(
    "--kernel_size",
    default=3,
    type=int,
    help="Kernel size for the unet.")
# Training parameters
@click.option(
    "--enhance_noise",
    default=False,
    type=bool,
    help="Enhance noise in training. Shift from a flat SNR distribution to a skewed distribution towards more noisy samples.")
@click.option(
    "--use_lr_scheduler",
    default=True,
    type=bool,
    help="Use learning rate scheduler.")
@click.option(
    "--n_shuffle",
    default=20,
    type=int,
    help="Shuffle number for the tensorflow datset.")
@click.option(
    "--batch_size",
    default=64,
    type=int,
    help="Training batch size.")
@click.option(
    "--n_epochs",
    default=500,
    type=int,
    help="Training number of epochs.")
@click.option(
    "--lr_param",
    default=1e-3,
    type=float,
    help="Learning rate.")
@click.option(
    "--data_train_ratio",
    default=0.9,
    type=float,
    help="Ratio of the dataset used for training.")
@click.option(
    "--snr_range",
    nargs=2,
    default=[1e-3, 50],
    type=float,
    help="SNR range for the added noise.")


def main(**args):
    print(args)
    mccd.script_utils.train_unets(**args)


if __name__ == "__main__":
  main()

