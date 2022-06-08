#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn

from datasets import load_dataset
from noise2noise import Noise2Noise
from argparse import ArgumentParser


def parse_args():
    """Command-line argument parser for training."""

    # New parser
    parser = ArgumentParser(description='AFM fast scan and noisy image reconstruction')

    # Data parameters
    parser.add_argument('-t', '--train-dir',
                        help='directory path containing training images',
                        default='../data/train')
    parser.add_argument('-v', '--valid-dir',
                        help='directory path containing validation images',
                        default='../data/valid')
    parser.add_argument('--target-dir',
                        help='directory path containing target images, if applicable.',
                        default='../data/targets')
    parser.add_argument('--ckpt-save-path', help='checkpoint save path', default='../ckpts')
    parser.add_argument('--ckpt-overwrite', help='overwrite model checkpoint on save', action='store_true')
    parser.add_argument('--report-interval', help='batch report interval', default=100, type=int)

    # Training hyperparameters
    parser.add_argument('-lr', '--learning-rate', help='learning rate', default=0.001, type=float)
    parser.add_argument('-a', '--adam', help='adam parameters', nargs='+', default=[0.9, 0.99, 1e-8], type=list)
    parser.add_argument('-b', '--batch-size', help='minibatch size', default=4, type=int)
    parser.add_argument('-e', '--nb-epochs', help='number of epochs', default=100, type=int)
    parser.add_argument('-l', '--loss', help='loss function', choices=['l1', 'l2'], default='l1', type=str)
    parser.add_argument('--cuda', help='use cuda', action='store_true')
    parser.add_argument('--plot-stats', help='plot stats after every epoch', action='store_true')

    # Corruption parameters
    parser.add_argument('-n', '--noise-type', help='noise type',
                        choices=['bernoulli', 'gradient', 'lower', 'nonuniform', 'raw'],
                        default='bernoulli', type=str)
    parser.add_argument('-p', '--noise-param', help='noise parameter', default=0.7, type=float)
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)
    parser.add_argument('-c', '--crop-size', help='random crop size', default=0, type=int)
    parser.add_argument('--clean-targets', help='use clean targets for training', action='store_true')
    parser.add_argument('--paired-targets', help='uses targets from "targets" folder in data directory', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    """Trains Noise2Noise."""

    # Parse training parameters
    params = parse_args()

    # debugging only
    params.train_dir = "../hs20mg_data/train"
    params.valid_dir = "../hs20mg_data/valid"
    params.target_dir = "../hs20mg_data/targets"
    params.ckpt_overwrite = True
    params.nb_epochs = 1
    params.noise_type = 'raw'
    params.paired_targets = True

    # Train/valid datasets
    train_loader = load_dataset(params.train_dir, params, shuffled=True)
    valid_loader = load_dataset(params.valid_dir, params, shuffled=False)

    # Initialize model and train
    n2n = Noise2Noise(params, trainable=True)
    n2n.train(train_loader, valid_loader)
