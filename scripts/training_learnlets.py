#!/usr/bin/env python
# coding: utf-8

import mccd
import click

@click.command()

## Training options
# Input parameters
@click.option(
    "--run_id_name",
    default="learnlet_256",
    type=str,
    help="Model id saving name.")
@click.option(
    "--dataset_path",
    default="/n05data/ayed/outputs/eigenpsfs/dataset_eigenpsfs.fits",
    type=str,
    help="Input dataset path.")
@click.option(
    "--im_shape",
    nargs=2,
    default=[51, 51],
    type=int,
    help="Images shape.")
# Saving paths
@click.option(
    "--base_save_path",
    default="/n05data/tliaudat/new_deepmccd/reproduce_aziz_results/trained_nets/learnlets_256/",
    type=str,
    help="Base path for saving files.")
# Learnlet parameters
@click.option(
    "--n_tiling",
    default=256,
    type=int,
    help="Number of filters for the Learnlet model.")
@click.option(
    "--n_scales",
    default=5,
    type=int,
    help="Number of scales for the Learnlet model.")
# Training parameters
@click.option(
    "--use_lr_scheduler",
    default=True,
    type=bool,
    help="Use learning rate scheduler.")
@click.option(
    "--enhance_noise",
    default=False,
    type=bool,
    help="Enhance noise in training. Shift from a flat SNR distribution to a skewed distribution towards more noisy samples.")
@click.option(
    "--n_shuffle",
    default=20,
    type=int,
    help="Shuffle number for the tensorflow datset.")
@click.option(
    "--batch_size",
    default=32,
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
    mccd.script_utils.train_learnlets(**args)


if __name__ == "__main__":
  main()
