#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os.path

import matplotlib
from matplotlib import rcParams
from torch.distributions.bernoulli import Bernoulli as BernoulliDist
from torch.distributions.normal import Normal as NormalDist
from torch.utils.data import Dataset, DataLoader
from random import shuffle

from data_prep import *
from utils import rescale_tensor, import_spm

rcParams['font.family'] = 'serif'
matplotlib.use('agg')


def load_dataset(root_dir, params, shuffled=False, single=False):
    """Loads dataset and returns corresponding data loader."""
    # Create Torch dataset
    noise = (params.noise_type, params.noise_param)

    dataset = NoisyDataset(root_dir, params.target_dir, params.redux, params.crop_size,
                           clean_targets=params.clean_targets, paired_targets=params.paired_targets,
                           channels=params.channels, noise_dist=noise, seed=params.seed, verbose=params.verbose)

    # Use batch size of 1, if requested (e.g. test set)
    if single:
        return DataLoader(dataset, batch_size=1, shuffle=shuffled)
    else:
        return DataLoader(dataset, batch_size=params.batch_size, shuffle=shuffled)


class NoisyDataset(Dataset):
    """Class for injecting random noise into dataset."""

    def __init__(self, root_dir, target_dir, redux=0, crop_size=0, clean_targets=False, paired_targets=False,
                 channels=3, noise_dist=('bernoulli', 0.7), seed=None, verbose=False):
        """Initializes noisy image dataset."""

        super(NoisyDataset, self).__init__()

        self.root_dir = os.path.normpath(root_dir)
        self.target_dir = os.path.normpath(target_dir)
        self.img_fnames = []
        self.trgt_fnames = []
        self.images = {}
        self.targets = {}
        self.redux = float(redux)
        self.crop_size = crop_size
        self.clean_targets = clean_targets
        self.paired_targets = paired_targets

        # Noise parameters
        self.noise_type = noise_dist[0]
        self.noise_param = noise_dist[1]
        self.seed = seed
        if self.seed:
            np.random.seed(self.seed)

        # load filenames and redux, if applicable
        ext_list = ['.png', '.jpeg', '.jpg', '.xyz', '.txt', '.csv']  # acceptable extensions/filetypes
        self.img_fnames = [s for s in os.listdir(root_dir) if os.path.splitext(s)[-1].lower() in ext_list]

        if self.paired_targets and not os.path.isdir(target_dir):
            raise NotADirectoryError(f"Paired targets are requested but the target directory ({target_dir}) does not exist!")

        if 0 < self.redux < 1:
            new_size = int((1 - self.redux) * len(self.img_fnames))
            shuffle(self.img_fnames)    # in-place shuffle of filenames so redux doesn't take the same files every time
            self.img_fnames = self.img_fnames[:new_size]  # reduce dataset size to given ratio
            # targets are found by source name anyway so no need to change targets list
        elif self.redux > 1:
            raise ValueError("redux ratio must be a float between 0.0 and 1.0 (non-inclusive)")

        # load images into memory
        if verbose:
            source_iter = tqdm(self.img_fnames, desc=f'Loading {os.path.basename(self.root_dir)} images', unit='img')
        else:
            source_iter = self.img_fnames
        self.images = {name: obj for (name, obj) in [import_spm(os.path.join(self.root_dir, s)) for s in source_iter]}

        # get targets in matching order to source list (if paired targets is true)
        if self.paired_targets:
            # self.trgt_fnames = [find_target(os.listdir(self.target_dir), s) for s in self.img_fnames]
            self.trgt_fnames = [s.replace('corrupt', 'clean') for s in self.img_fnames]     # find_target is taking wayyy too long
            if verbose:
                target_iter = tqdm(self.trgt_fnames, desc=f'Loading {os.path.basename(self.root_dir)} targets', unit='img')
            else:
                target_iter = self.trgt_fnames
            self.targets = {name: obj for (name, obj) in [import_spm(os.path.join(self.target_dir, t)) for t in target_iter]}

    def _add_noise(self, img: torch.Tensor, param: float) -> torch.Tensor:
        """ Adds noise to an image. """

        rng = np.random.default_rng()
        c, h, w = img.shape
        valid_types = ['bernoulli', 'gradient', 'lower', 'gaussian']

        if self.noise_type.lower() not in valid_types:
            raise ValueError(f'Invalid noise type: {self.noise_type}. Added noise must be one of {valid_types}.')
        
        gen = torch.Generator()
        if self.seed:
            gen.manual_seed(self.seed)

        if self.noise_type.lower() == 'gaussian':
            std = rng.uniform(0, param) * img.std()
            noisy_img = img.to(torch.float64, copy=True) + NormalDist(0, std, generator=gen).sample(img.size())
            if not img.is_floating_point():     # assumes int type means it's an image format (e.g., 0-255 range), so need rescale after summing
                noisy_img = rescale_tensor(noisy_img, bounds=[img.min().item(), img.max().item()]).to(img.dtype)

        elif self.noise_type.lower() == 'lower':
            hN, wN = int(h * param), int(w * param)
            noisy_img = tvF.resize(tvF.resize(img.to(torch.float64, copy=True), [hN, wN]), [h, w])

        elif self.noise_type.lower() == 'gradient':
            std = param * img.std()
            noise = torch.from_numpy(np.tile(np.linspace(0, 1, w), (h, 1))).unsqueeze(dim=0) * NormalDist(0, std).sample(img.size())
            noisy_img = img.to(torch.float64, copy=True) + noise
            if not img.is_floating_point():     # assumes int type means it's an image format (e.g., 0-255 range), so need rescale after summing
                noisy_img = rescale_tensor(noisy_img, bounds=[img.min().item(), img.max().item()]).to(img.dtype)

        else:   # self.noise_type.lower() == 'bernoulli':
            noisy_img = img.to(torch.float64, copy=True) * BernoulliDist(param).sample(img.size()[1:])

        return noisy_img

    def __getitem__(self, index):
        """Retrieves image from folder_path and corrupts it."""

        # Load image
        ground_truth = self.images[self.img_fnames[index]]

        if self.noise_type != 'raw':
            if self.noise_param == 0:
                param = np.random.choice(np.arange(.1, .9, .05))
            else:
                param = self.noise_param

            # Corrupt source image
            source = self._add_noise(ground_truth, param)

            # Corrupt target image, but not when clean targets are requested or pairs exist
            if self.paired_targets:  # paired targets overrides clean targets
                target = self.targets[self.trgt_fnames[index]]
            elif self.clean_targets:
                target = ground_truth.clone()
            else:
                target = self._add_noise(ground_truth, param)

        else:  # self.noise_type == 'raw':
            # if noise type is raw, assume paired targets
            source = ground_truth.clone()
            target = self.targets[self.trgt_fnames[index]]

        if self.crop_size > 0 and self.crop_size < min(self.images[0].shape[1:]):
            top = random.choice(range(self.images[self.img_fnames[index]].shape[1] - self.crop_size))
            left = random.choice(range(self.images[self.img_fnames[index]].shape[2] - self.crop_size))
            source = tvF.crop(source, top, left, self.crop_size, self.crop_size)
            target = tvF.crop(target, top, left, self.crop_size, self.crop_size)

        return source.to(torch.float32), target.to(torch.float32)       # temporary fix for bug in pytorch... see https://github.com/pytorch/pytorch/issues/111671 (Jan 2024)

    def __len__(self):
        """Returns length of dataset."""

        return len(self.img_fnames)
