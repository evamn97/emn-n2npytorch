#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from PIL import Image
from matplotlib import rcParams

from torch.utils.data import Dataset, DataLoader

rcParams['font.family'] = 'serif'
matplotlib.use('agg')


def create_image(img, p=0.5, style='r'):
    """ Creates noisy image for source or target."""
    ground_truth_array = np.array(img)
    if len(ground_truth_array.shape) == 3:
        w, h, c = ground_truth_array.shape  # get  array dimensions
    else:
        w = ground_truth_array.shape[0]
        h = ground_truth_array.shape[1]
        c = 1

    rng = np.random.default_rng()

    if style == 'l':      # lowers resolution of input image (should be used with clean targets)
        temp_img = img
        hN, wN = int(h*p), int(w*p)     # choose new dims based on p value
        resized = tvF.resize(tvF.resize(temp_img, [hN, wN]), [h, w])
        img_array_noised = np.array(resized)

    elif style == 'nu':     # random noise up to +/- 10 percent values (not binary)
        ran = (np.max(ground_truth_array) - np.min(ground_truth_array))*0.1
        rand_noise = rng.uniform(-1*ran, ran, (h, w))
        bin_mask = rng.binomial(1, p, (h, w))
        mask = np.multiply(rand_noise, bin_mask)
        if c > 1:
            m_list = []
            for dim in range(c):
                m_list.append(mask)
            mask = np.dstack(m_list)
        img_array_noised = np.add(ground_truth_array, mask)

    elif style == 'g':     # create a noise gradient from left to right
        ran = (np.max(ground_truth_array) - np.min(ground_truth_array))*p     # use 0.1 here for +/- 10% noise; this is arbitrary
        rand_noise = rng.uniform(-1, 1, (h, w))
        grad = np.tile(np.linspace(0, ran, num=w), (h, 1))
        mask = np.multiply(rand_noise, grad)
        if c > 1:
            m_list = []
            for dim in range(c):
                m_list.append(mask)
            mask = np.dstack(m_list)
        img_array_noised = np.add(ground_truth_array, mask)

    else:   # style == 'r':  # random trials
        r_mask = rng.binomial(1, p, (h, w))
        if c > 1:
            m_list = []
            for dim in range(c):
                m_list.append(r_mask)
            mask = np.dstack(m_list)
        else:
            mask = r_mask
        img_array_noised = np.multiply(ground_truth_array, mask)

    return img_array_noised


def load_dataset(root_dir, params, shuffled=False, single=False):
    """Loads dataset and returns corresponding data loader."""

    # Create Torch dataset
    noise = (params.noise_type, params.noise_param)

    dataset = NoisyDataset(root_dir, params.crop_size,
                           clean_targets=params.clean_targets, noise_dist=noise, seed=params.seed)

    # Use batch size of 1, if requested (e.g. test set)
    if single:
        return DataLoader(dataset, batch_size=1, shuffle=shuffled)
    else:
        return DataLoader(dataset, batch_size=params.batch_size, shuffle=shuffled)


class AbstractDataset(Dataset):
    """Abstract dataset class for Noise2Noise."""

    def __init__(self, root_dir, crop_size=128, clean_targets=False, paired_targets=False):
        """Initializes abstract dataset."""

        super(AbstractDataset, self).__init__()

        self.imgs = []
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.clean_targets = clean_targets
        self.paired_targets = paired_targets
        if self.paired_targets:
            # self.targets = []     # TODO: Remove this? I think I can just retrieve by name from source
            self.target_dir = os.path.join(os.path.dirname(root_dir), "targets")    # if target pairs exist, get dir

    def _random_crop(self, img_list):
        """Performs random square crop of fixed size.
        Works with list so that all items get the same cropped window (e.g. for buffers).
        """

        w, h = img_list[0].size
        assert w >= self.crop_size and h >= self.crop_size, \
            f'Error: Crop size: {self.crop_size}, Image size: ({w}, {h})'  # this assertion is redundant with resizing below
        cropped_imgs = []
        i = np.random.randint(0, h - self.crop_size + 1)
        j = np.random.randint(0, w - self.crop_size + 1)

        for img in img_list:
            # Resize if dimensions are too small
            if min(w, h) < self.crop_size:
                img = tvF.resize(img, [self.crop_size, self.crop_size])

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

    def __init__(self, root_dir, crop_size, clean_targets=False, paired_targets=False,
                 noise_dist=('bernoulli', 0.7), seed=None):
        """Initializes noisy image dataset."""

        super(NoisyDataset, self).__init__(root_dir, crop_size, clean_targets, paired_targets)

        self.imgs = []
        with os.scandir(root_dir) as folder:
            for item in folder:
                if any(ext in item.name.lower() for ext in ['.png', '.jpeg', '.jpg']):
                    self.imgs.append(item.name)

        # if self.paired_targets:
        #     # TODO: remove this? I don't think I need a list of targets, don't want to rely on sort
        #     self.targets = []   # not used for now, more reliable to get it from name so order doesn't matter
        #     with os.scandir(self.target_dir) as folder:
        #         for item in folder:
        #             if any(ext in item.name.lower() for ext in ['.png', '.jpeg', '.jpg']):
        #                 self.targets.append(item.name)

        # Noise parameters
        self.noise_type = noise_dist[0]
        self.noise_param = noise_dist[1]
        self.seed = seed
        if self.seed:
            np.random.seed(self.seed)

    def _add_noise(self, img):
        """Adds noise to image."""

        w, h = img.size
        c = len(img.getbands())

        # Manipulated noise options
        if self.noise_type == 'gradient':
            noise_img = create_image(img, p=self.noise_param, style='g')
        elif self.noise_type == 'lower':
            noise_img = create_image(img, p=self.noise_param, style='l')
        elif self.noise_type == 'nonuniform':
            noise_img = create_image(img, p=self.noise_param, style='nu')

        # Bernoulli distribution (default)
        else:
            #  self.noise_type == 'bernoulli':
            noise_img = create_image(img, p=self.noise_param, style='r')

        noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
        return Image.fromarray(noise_img)

    def _corrupt(self, img):
        """Corrupts images."""

        if self.noise_type in ['bernoulli', 'gradient', 'lower', 'nonuniform']:
            return self._add_noise(img)
        elif self.noise_type in ['raw']:    # for input/target paired training
            return img
        else:
            raise ValueError('Invalid noise type: {}'.format(self.noise_type))

    def __getitem__(self, index):
        """Retrieves image from folder_path and corrupts it."""

        # Load PIL image
        img_path = os.path.join(self.root_dir, self.imgs[index])
        with Image.open(img_path).convert('RGB') as img:
            img.load()

        if self.paired_targets:
            trgt_name = "target_" + self.imgs[index]  # get target from name instead of relying on sorting
            trgt_path = os.path.join(self.target_dir, trgt_name)
            with Image.open(trgt_path).convert('RGB') as trgt:
                trgt.load()

        # Random square crop
        if not self.paired_targets and self.crop_size != 0:
            img = self._random_crop([img])[0]

        # Corrupt source image
        tmp = self._corrupt(img)    # see '_corrupt' for returning raw image option
        source = tvF.to_tensor(self._corrupt(img))

        # Corrupt target image, but not when clean targets are requested or pairs exist
        if self.clean_targets and not self.paired_targets:
            target = tvF.to_tensor(img)
        elif self.paired_targets:
            target = tvF.to_tensor(trgt)
        else:
            target = tvF.to_tensor(self._corrupt(img))

        return source, target
