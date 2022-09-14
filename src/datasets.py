#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as trf
import torchvision.transforms.functional as tvF
from PIL import Image
from matplotlib import rcParams

from data_prep import *

from torch.utils.data import Dataset, DataLoader

rcParams['font.family'] = 'serif'
matplotlib.use('agg')


def create_image(img, p=0.5, style='r'):
    """ Creates noisy image for source or target."""
    ground_truth_array = np.array(img)
    if len(ground_truth_array.shape) == 3:  # get  array dimensions for RGB and RGBA images
        w = ground_truth_array.shape[0]
        h = ground_truth_array.shape[1]
        c = ground_truth_array.shape[2]
    else:  # get dims for grayscale images/height fields
        w = ground_truth_array.shape[0]
        h = ground_truth_array.shape[1]
        c = 1

    rng = np.random.default_rng()

    if style == 'l':  # lowers resolution of input image (should be used with clean targets)
        temp_img = img
        hN, wN = int(h * p), int(w * p)  # choose new dims based on p value
        resized = tvF.resize(tvF.resize(temp_img, [hN, wN]), [h, w])
        img_array_noised = np.array(resized)

    # if style == 'l':  # lowers resolution of input image (editing for xyz images as well - doesn't work yet)
    #     temp_img = tvF.to_tensor(img)
    #     hN, wN = int(h * p), int(w * p)  # choose new dims based on p value
    #     resized = tvF.resize(tvF.resize(temp_img, [hN, wN]), [h, w])
    #     if c == 1:
    #         img_array_noised = np.array(resized.numpy().squeeze())
    #     else:
    #         m_list = []
    #         for dim in range(c):
    #             m_list.append(resized[dim])
    #         img_array_noised = np.dstack(m_list)

    elif style == 'nu':  # random noise up to +/- 10 percent values (not binary)
        ran = (np.max(ground_truth_array) - np.min(ground_truth_array)) * 0.1
        rand_noise = rng.uniform(-1 * ran, ran, (h, w))
        bin_mask = rng.binomial(1, p, (h, w))
        mask = np.multiply(rand_noise, bin_mask)
        if c > 1:
            m_list = []
            for dim in range(c):
                m_list.append(mask)
            mask = np.dstack(m_list)
        img_array_noised = np.add(ground_truth_array, mask)

    elif style == 'g':  # create a noise gradient from left to right
        ran = (np.max(ground_truth_array) - np.min(ground_truth_array)) * p  # use 0.1 here for +/- 10% noise; this is arbitrary
        rand_noise = rng.uniform(-1, 1, (h, w))
        grad = np.tile(np.linspace(0, ran, num=w), (h, 1))
        mask = np.multiply(rand_noise, grad)
        if c > 1:
            m_list = []
            for dim in range(c):
                m_list.append(mask)
            mask = np.dstack(m_list)
        img_array_noised = np.add(ground_truth_array, mask)

    else:  # style == 'r':  # random trials
        r_mask = rng.binomial(1, p, (h, w))
        if c > 1:
            m_list = []
            for dim in range(c):
                m_list.append(r_mask)
            mask = np.dstack(m_list)
        else:
            mask = r_mask
        img_array_noised = np.multiply(ground_truth_array, mask)

    if c == 1:
        final_noised = normalize(img_array_noised)
    else:  # c == 3
        final_noised = np.clip(img_array_noised, 0, 255).astype(np.uint8)
    return final_noised


def load_dataset(root_dir, params, shuffled=False, single=False):
    """Loads dataset and returns corresponding data loader."""
    # Create Torch dataset
    noise = (params.noise_type, params.noise_param)

    dataset = NoisyDataset(root_dir, params.target_dir, params.redux, params.crop_size,
                           clean_targets=params.clean_targets, paired_targets=params.paired_targets,
                           channels=params.channels, noise_dist=noise, seed=params.seed)

    # Use batch size of 1, if requested (e.g. test set)
    if single:
        return DataLoader(dataset, batch_size=1, shuffle=shuffled)
    else:
        return DataLoader(dataset, batch_size=params.batch_size, shuffle=shuffled)


class AbstractDataset(Dataset):
    """Abstract dataset class for Noise2Noise."""

    def __init__(self, root_dir, target_dir, redux=0, crop_size=0, clean_targets=False, paired_targets=False, channels=3):
        """Initializes abstract dataset."""

        super(AbstractDataset, self).__init__()

        self.imgs = []
        self.targets = []
        self.root_dir = root_dir
        self.redux = float(redux)
        self.crop_size = crop_size
        self.clean_targets = clean_targets
        self.paired_targets = paired_targets
        self.target_dir = target_dir
        self.channels = channels

    def _random_crop(self, img_list):
        """Performs random square crop of fixed size.
        Works with list so that all items get the same cropped window (e.g. for buffers).
        """
        cropped_imgs = []

        for img in img_list:
            if type(img) != torch.Tensor:
                im = tvF.to_tensor(img)
            else:
                im = img
            c, h, w = im.size()
            # Resize if dimensions are too small
            if min(w, h) < self.crop_size:
                im = tvF.resize(img, [self.crop_size, self.crop_size])
            cropped_im = trf.RandomCrop(self.crop_size)(im).numpy().squeeze()
            if c > 1:
                cropped_im = np.rot90(np.rot90(cropped_im, axes=(0, 2)), k=3)
            # Random crop
            cropped_imgs.append(cropped_im)  # numpy arrays with shape (h, w, c)

        return cropped_imgs

    def __getitem__(self, index):
        """Retrieves image from data folder path."""

        raise NotImplementedError('Abstract method not implemented!')

    def __len__(self):
        """Returns length of dataset."""

        return len(self.imgs)


class NoisyDataset(AbstractDataset):
    """Class for injecting random noise into dataset."""

    def __init__(self, root_dir, target_dir, redux, crop_size, clean_targets=False, paired_targets=False, channels=3,
                 noise_dist=('bernoulli', 0.7), seed=None):
        """Initializes noisy image dataset."""

        super(NoisyDataset, self).__init__(root_dir, target_dir, redux, crop_size, clean_targets, paired_targets, channels)

        ext_list = ['.png', '.jpeg', '.jpg', '.xyz', '.txt', '.csv']  # acceptable extensions/filetypes
        self.imgs = [s for s in os.listdir(root_dir) if os.path.splitext(s)[-1].lower() in ext_list]
        if os.path.isdir(target_dir):
            self.targets = [t for t in os.listdir(target_dir) if os.path.splitext(t)[-1].lower() in ext_list]
        if self.paired_targets and not os.path.isdir(target_dir):
            raise NotADirectoryError("Paired targets are requested but the input target directory does not exist!")

        if self.redux != 0:
            if not 0.0 < self.redux < 1.0:
                raise ValueError("redux ratio must be a float between 0.0 and 1.0 (non-inclusive)")
            new_size = int(self.redux * len(self.imgs))
            self.imgs = self.imgs[:new_size]  # reduce dataset size to given ratio
            # targets are found by source name anyway so no need to change self.targets list

        # Noise parameters
        self.noise_type = noise_dist[0]
        self.noise_param = noise_dist[1]
        self.seed = seed
        if self.seed:
            np.random.seed(self.seed)

    def _add_noise(self, img):
        """Adds noise to image."""

        # Manipulated noise options
        if self.noise_type == 'gradient':
            noise_img = create_image(img, p=self.noise_param, style='g')
        elif self.noise_type == 'lower':
            noise_img = create_image(img, p=self.noise_param, style='l')
        elif self.noise_type == 'nonuniform':
            noise_img = create_image(img, p=self.noise_param, style='nu')
        elif self.noise_type == 'raw':  # just for redundancy & debugging
            noise_img = np.array(img)

        # Bernoulli distribution (default)
        else:  # self.noise_type == 'bernoulli':
            noise_img = create_image(img, p=self.noise_param, style='r')

        return noise_img  # returns numpy array in shape (h, w, c)

    def _corrupt(self, img):
        """Corrupts images."""

        if self.noise_type in ['bernoulli', 'gradient', 'lower', 'nonuniform', 'raw']:
            return self._add_noise(img)  # numpy array (h, w, c)
        else:
            raise ValueError('Invalid noise type: {}'.format(self.noise_type))

    def __getitem__(self, index):
        """Retrieves image from folder_path and corrupts it."""

        # Load image
        img_name = self.imgs[index]
        img_path = os.path.normpath(os.path.join(self.root_dir, self.imgs[index]))

        if os.path.splitext(img_name)[-1] in ['.xyz', '.txt', '.csv']:  # load xyz
            img = normalize(xyz_to_zfield(img_path, return3d=False))
            if self.channels != 1:
                raise ValueError("The number of channels for xyz files must be 1, but got {}. Check channels input.".format(self.channels))
        else:  # load image
            with Image.open(img_path).convert('RGB') as img:
                img.load()
                if self.channels != 3:
                    raise ValueError("The number of channels for image files must be 3, but got {}. Check channels input.".format(self.channels))
        # Random square crop
        if not self.paired_targets and self.crop_size > 0:
            img = self._random_crop([img])[0]
        # Corrupt source image
        source = tvF.to_tensor(self._corrupt(img))  # see '_corrupt' for returning raw image option

        # Corrupt target image, but not when clean targets are requested or pairs exist
        if self.paired_targets:  # paired targets overrides clean targets
            trgt_name = find_target(self.targets, self.imgs[index])
            trgt_path = os.path.normpath(os.path.join(self.target_dir, trgt_name))
            if os.path.splitext(trgt_name)[-1] in ['.xyz', '.txt', '.csv']:
                trgt = normalize(xyz_to_zfield(trgt_path, return3d=False))
            else:
                with Image.open(trgt_path).convert('RGB') as trgt:
                    trgt.load()
            target = tvF.to_tensor(trgt)
        elif self.clean_targets:
            target = tvF.to_tensor(img)
        else:
            target = tvF.to_tensor(self._corrupt(img))

        return source, target
