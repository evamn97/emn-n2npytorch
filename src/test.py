#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser

from datasets import load_dataset
from noise2noise import Noise2Noise


def parse_args():
    """Command-line argument parser for testing."""

    # New parser
    parser = ArgumentParser(description='AFM fast scan and noisy image reconstruction')

    # Data parameters
    parser.add_argument('-t', '--test-dir', help='directory path containing testing images', default='../data/test')
    parser.add_argument('--target-dir', help='directory path containing target images, if applicable.', default='../data/targets')
    parser.add_argument('-r', '--redux', help='ratio (0.1 - 0.99) of dataset size to use for training (redux=0 means no reduction). useful for quick debugging.', default=0)
    parser.add_argument('--output', help='directory to save the results images', default='../results')
    parser.add_argument('--load-ckpt', help='load model checkpoint')
    parser.add_argument('--show-output', help='pop up window to display outputs', default=0, type=int)
    parser.add_argument('--montage-only', help='save montage figures only (not inputs and outputs)', action='store_true')
    parser.add_argument('--cuda', help='use cuda', action='store_true')

    # Corruption parameters
    parser.add_argument('-n', '--noise-type', help='noise type',
                        choices=['bernoulli', 'gradient', 'lower', 'nonuniform', 'raw'], default='bernoulli', type=str)
    parser.add_argument('-p', '--noise-param', help='noise parameter', default=0.7, type=float)
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)
    parser.add_argument('-c', '--crop-size', help='image crop size', default=0, type=int)
    parser.add_argument('-ch', '--channels', help='change the number of input/output channels for Unet (ex: RGB=3, L=1, LA=2)', default=3, type=int)
    parser.add_argument('--paired-targets', help='uses targets from "targets" directory', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    """Tests Noise2Noise."""

    # Parse test parameters
    params = parse_args()

    # error handling for noise param greater than 1
    if params.noise_param >= 1:
        mag = len(str(int(params.noise_param)))
        params.noise_param = params.noise_param / (10 ** mag)

    # debugging only
    # parent = '../hs20mg_xyz_data'
    # params.test_dir = os.path.join(parent, 'test')
    # params.target_dir = os.path.join(params.test_dir, 'targets')
    # params.load_ckpt = '../ckpts_new/hs20mg-xyz-raw/n2n-epoch100-0.00000.pt'
    # params.montage_only = True
    # params.noise_type = 'raw'
    # # params.noise_param = 0.4
    # params.paired_targets = True
    # params.channels = 1
    # params.output = '../new_results/'

    # Initialize model and test
    n2n = Noise2Noise(params, trainable=False)
    params.clean_targets = True
    test_loader = load_dataset(params.test_dir, params, shuffled=False, single=True)
    n2n.load_model(params.load_ckpt)
    n2n.test(test_loader, show=params.show_output)
