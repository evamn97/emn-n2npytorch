#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from torch.utils.data import Dataset, DataLoader

import os
from sys import platform
import numpy as np
import random
from string import ascii_letters
from PIL import Image, ImageFont, ImageDraw
import OpenEXR

from matplotlib import rcParams

import bern_utils_PIL as b


rcParams['font.family'] = 'serif'
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt


def load_dataset(root_dir, redux, params, shuffled=False, single=False):
    """Loads dataset and returns corresponding data loader."""

    # Create Torch dataset
    noise = (params.noise_type, params.noise_param)
    
    dataset = NoisyDataset(root_dir, redux, params.crop_size,
                           clean_targets=params.clean_targets, noise_dist=noise, seed=params.seed)

    # Use batch size of 1, if requested (e.g. test set)
    if single:
        return DataLoader(dataset, batch_size=1, shuffle=shuffled)
    else:
        return DataLoader(dataset, batch_size=params.batch_size, shuffle=shuffled)


class AbstractDataset(Dataset):
    """Abstract dataset class for Noise2Noise."""

    def __init__(self, root_dir, redux=0, crop_size=128, clean_targets=False):
        """Initializes abstract dataset."""

        super(AbstractDataset, self).__init__()

        self.imgs = []
        self.root_dir = root_dir
        self.redux = redux
        self.crop_size = crop_size
        self.clean_targets = clean_targets

    def _random_crop(self, img_list):
        """Performs random square crop of fixed size.
        Works with list so that all items get the same cropped window (e.g. for buffers).
        """

        w, h = img_list[0].size
        assert w >= self.crop_size and h >= self.crop_size, \
            f'Error: Crop size: {self.crop_size}, Image size: ({w}, {h})'   # this assertion is redundant with resizing below
        cropped_imgs = []
        i = np.random.randint(0, h - self.crop_size + 1)
        j = np.random.randint(0, w - self.crop_size + 1)

        for img in img_list:
            # Resize if dimensions are too small
            if min(w, h) < self.crop_size:
                img = tvF.resize(img, (self.crop_size, self.crop_size))

            # Random crop
            cropped_imgs.append(tvF.crop(img, i, j, self.crop_size, self.crop_size))

        return cropped_imgs

    def __getitem__(self, index):
        """Retrieves image from data folder_path."""

        raise NotImplementedError('Abstract method not implemented!')

    def __len__(self):
        """Returns length of dataset."""

        return len(self.imgs)


class NoisyDataset(AbstractDataset):
    """Class for injecting random noise into dataset."""

    def __init__(self, root_dir, redux, crop_size, clean_targets=False,
                 noise_dist=('bernoulli', 0.7), seed=None):
        """Initializes noisy image dataset."""

        super(NoisyDataset, self).__init__(root_dir, redux, crop_size, clean_targets)

        self.imgs = []
        with os.scandir(root_dir) as folder:
            for item in folder:
                if any(ext in item.name.lower() for ext in ['.png', '.jpeg', '.jpg']):
                    self.imgs.append(item.name)
        if redux:
            self.imgs = self.imgs[:redux]

        # Noise parameters
        self.noise_type = noise_dist[0]
        self.noise_param = noise_dist[1]
        self.seed = seed
        if self.seed:
            np.random.seed(self.seed)

    def _add_noise(self, img):
        """Adds Bernoulli noise to image."""

        w, h = img.size
        c = len(img.getbands())

        # Manipulated binary noise options
        if self.noise_type == 'gradient':
            noise_img = b.create_image(img, p=self.noise_param, style='g')
        elif self.noise_type == 'lower':
            noise_img = b.create_image(img, p=self.noise_param, style='l')
        elif self.noise_type == 'nonuniform':
            noise_img = b.create_image(img, p=self.noise_param, style='nu')

        # Bernoulli distribution (default)
        else:
            #  self.noise_type == 'bernoulli':
            noise_img = b.create_image(img, p=self.noise_param, style='r')

        noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
        return Image.fromarray(noise_img)


    def _corrupt(self, img):
        """Corrupts images (Bernoulli)."""

        if self.noise_type in ['bernoulli', 'gradient', 'lower', 'nonuniform']:
            return self._add_noise(img)
        else:
            raise ValueError('Invalid noise type: {}'.format(self.noise_type))

    def __getitem__(self, index):
        """Retrieves image from folder_path and corrupts it."""

        # Load PIL image
        img_path = os.path.join(self.root_dir, self.imgs[index])

        img = Image.open(img_path).convert('RGB')

        # Random square crop
        if self.crop_size != 0:
            img = self._random_crop([img])[0]

        # Corrupt source image
        tmp = self._corrupt(img)
        source = tvF.to_tensor(self._corrupt(img))

        # Corrupt target image, but not when clean targets are requested
        if self.clean_targets:
            target = tvF.to_tensor(img)
        else:
            target = tvF.to_tensor(self._corrupt(img))

        return source, target
