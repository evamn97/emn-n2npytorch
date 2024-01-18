#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os
from argparse import ArgumentParser

from datasets import load_dataset
from noise2noise import Noise2Noise


def parse_args():
    """Command-line argument parser for training."""

    # New parser
    parser = ArgumentParser(description='AFM fast scan and noisy image reconstruction')

    # Data parameters
    parser.add_argument('-t', '--train-dir', help='directory path containing training images', default='../data/train', type=str)
    parser.add_argument('-v', '--valid-dir', help='directory path containing validation images', default='../data/valid', type=str)
    parser.add_argument('--target-dir', help='directory path containing target images, if applicable.',
                        default='../data/targets', type=str)
    parser.add_argument('-r', '--redux',
                        help='reduction ratio [0, 1) of dataset size (redux=0 means no reduction). useful for quick debugging.',
                        default=0, type=float)
    parser.add_argument('--load-ckpt', help='load ckpt from a previously trained model to use in training',
                        default=None, type=str)
    parser.add_argument('--ckpt-save-path', help='checkpoint save path', default='../ckpts', type=str)
    parser.add_argument('--ckpt-overwrite', help='overwrite model checkpoint on save', action='store_true')
    parser.add_argument('--report-interval', help='batch report interval', default=100, type=int)

    # Training hyperparameters
    parser.add_argument('-lr', '--learning-rate', help='learning rate', default=0.001, type=float)
    parser.add_argument('-a', '--adam', help='adam parameters', nargs='+', default=[0.9, 0.99, 1e-8], type=list)
    parser.add_argument('-ch', '--channels', help='change the number of input/output channels for Unet (ex: RGB=3, L=1, LA=2)', default=3, type=int)
    parser.add_argument('-b', '--batch-size', help='minibatch size', default=4, type=int)
    parser.add_argument('-e', '--nb-epochs', help='number of epochs', default=100, type=int)
    parser.add_argument('-l', '--loss', help='loss function', choices=['l1', 'l2'], default='l1', type=str)
    parser.add_argument('--cuda', help='use cuda', action='store_true')
    parser.add_argument('--verbose', help='prints training stats at report intervals', action='store_true')
    parser.add_argument('--show-progress', help='extra verbose: shows progress bar during training epochs in addition to training stats', action='store_true')

    # Corruption parameters
    parser.add_argument('-n', '--noise-type', help='noise type',
                        choices=['bernoulli', 'gradient', 'lower', 'gaussian', 'raw'], default='bernoulli', type=str)
    parser.add_argument('-p', '--noise-param', help='noise parameter [0, 1). set to 0 for random', default=0.7, type=float)
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)
    parser.add_argument('-c', '--crop-size', help='random crop size', default=0, type=int)
    parser.add_argument('--clean-targets', help='use clean targets for training', action='store_true')
    parser.add_argument('--paired-targets', help='uses targets from "targets" folder in data directory. overrides "--clean-targets"', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    """Trains Noise2Noise."""

    python_start = datetime.datetime.now()
    local_tz = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
    print(f'python start time:  {python_start.strftime("%H:%M:%S.%f")[:-4]} {local_tz}')

    # Parse training parameters
    params = parse_args()

    # debugging!! ------------------------------------------------------------------------------
    # root = "/Users/emnatin/Library/CloudStorage/OneDrive-SandiaNationalLaboratories/GitHub/"
    # root = "/mnt/data/emnatin"
    # parent = os.path.join(root, "timgrec-extra-tiny-ImageNet")
    # parent = os.path.join(root, "imgrec-tiny-ImageNet")
    # parent = "../hs20mg_xyz_data"
    # params.train_dir = os.path.join(parent, "train")
    # params.valid_dir = os.path.join(parent, "valid")
    # params.target_dir = os.path.join(parent, "targets")
    # params.batch_size = 4
    # params.nb_epochs = 10
    # params.channels = 1
    # params.loss = 'l1'
    # params.cuda = True
    # params.verbose = True
    # params.noise_type = 'raw'
    # params.paired_targets = True
    # ------------------------------------------------------------------------------------------

    if (params.noise_type == 'raw' and not params.paired_targets):
        params.paired_targets = True
    if params.paired_targets:
        params.clean_targets = False

    # error handling for noise param greater than 1 or random vs fixed
    if params.noise_param >= 1:
        mag = len(str(int(params.noise_param)))
        params.noise_param = params.noise_param / (10 ** mag)

    if params.show_progress:
        params.verbose = True  # so '--show-progress' can be used instead of '--verbose'

    # Train/valid datasets
    train_loader = load_dataset(params.train_dir, params, shuffled=True)
    valid_loader = load_dataset(params.valid_dir, params, shuffled=False)

    # Initialize model and train
    n2n = Noise2Noise(params, trainable=True)

    # try using previously trained model to build on - 06/17/22
    if params.load_ckpt and os.path.isfile(params.load_ckpt):
        print("\nLoading previous training model checkpoint...")
        n2n.load_model(params.load_ckpt)
    elif params.load_ckpt and not os.path.isfile(params.load_ckpt):
        # print("\nRequested model checkpoint ({}) is not a file. \nCreating a new training checkpoint.\n".format(params.load_ckpt))
        params.load_ckpt = None

    # print(f'training begin:      {str(datetime.datetime.now() - python_start)[:-4]} from python start')
    print(f'training begin:      {datetime.datetime.now().strftime("%H:%M:%S.%f")[:-4]}')

    n2n.train(train_loader, valid_loader)
