#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from datasets import load_dataset
from noise2noise import Noise2Noise

from argparse import ArgumentParser


def parse_args():
    """Command-line argument parser for testing."""

    # New parser
    parser = ArgumentParser(description='AFM fast scan and noisy image reconstruction')

    # Data parameters
    parser.add_argument('-t', '--test-dir', help='directory path containing testing images', default='../data/test')
    parser.add_argument('--target-dir',
                        help='directory path containing target images, if applicable.',
                        default='../data/targets')
    parser.add_argument('--output', help='directory to save the results images', default='../results')
    parser.add_argument('--load-ckpt', help='load model checkpoint')
    parser.add_argument('--show-output', help='pop up window to display outputs', default=0, type=int)
    parser.add_argument('--montage-only', help='save montage figures only (not inputs and outputs)', action='store_true')
    parser.add_argument('--cuda', help='use cuda', action='store_true')

    # Corruption parameters
    parser.add_argument('-n', '--noise-type', help='noise type',
                        choices=['bernoulli', 'gradient', 'lower', 'nonuniform', 'raw'],
                        default='bernoulli', type=str)
    parser.add_argument('-p', '--noise-param', help='noise parameter', default=0.7, type=float)
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)
    parser.add_argument('-c', '--crop-size', help='image crop size', default=0, type=int)
    parser.add_argument('--paired-targets', help='uses targets from "targets" directory', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    """Tests Noise2Noise."""

    # Parse test parameters
    params = parse_args()

    # Initialize model and test
    n2n = Noise2Noise(params, trainable=False)
    params.clean_targets = True
    test_loader = load_dataset(params.test_dir, params, shuffled=False, single=True)
    n2n.load_model(params.load_ckpt)
    n2n.test(test_loader, show=params.show_output)
