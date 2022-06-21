#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler

from unet import UNet
from utils import *

import os
import json


class Noise2Noise(object):
    """Implementation of Noise2Noise from Lehtinen et al. (2018)."""

    def __init__(self, params, trainable):
        """Initializes model."""

        self.p = params
        self.trainable = trainable
        self._compile()  # creates unet

        # try to sync montage output with SLURM output filenames by getting job ID and name - emn 05/19/22
        if "jobid" in os.environ:
            self.job_id = os.environ["jobid"]  # ex: 193174
        else:
            self.job_id = f'{datetime.now():%m%d%H%M}'  # ex: 05311336
        if "jobname" in os.environ and "idv" not in os.environ["jobname"]:
            self.job_name = os.environ["jobname"] + "-" + self.p.noise_type  # ex: 01-n2npt-train-bernoulli
        elif "filename" in os.environ:
            self.job_name = os.environ["filename"] + "-" + self.p.noise_type  # ex: debug-n2npt-train-bernoulli (ext removed)
        else:
            self.job_name = self.p.noise_type  # ex: bernoulli
            if trainable and self.p.clean_targets:
                self.job_name += "-clean"  # ex: bernoulli-clean
            if self.p.paired_targets:
                self.job_name += "-paired"  # ex: bernoulli-paired
            self.job_name += "-" + self.job_id  # ex: bernoulli-paired-05311336

    def _compile(self):
        """Compiles model (architecture, loss function, optimizers, etc.)."""

        # Model
        self.model = UNet(in_channels=self.p.channels, out_channels=self.p.channels).double()  # changed this from default in_channels=3 6/13/22 emn

        # Set optimizer and loss, if in training mode
        if self.trainable:
            self.optim = Adam(self.model.parameters(),
                              lr=self.p.learning_rate,
                              betas=self.p.adam[:2],
                              eps=self.p.adam[2])

            # Learning rate adjustment
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim,
                                                            patience=self.p.nb_epochs / 4, factor=0.5, verbose=True)

            # Loss function
            if self.p.loss == 'l2':
                self.loss = nn.MSELoss()
            else:
                self.loss = nn.L1Loss()

        # CUDA support
        self.use_cuda = torch.cuda.is_available() and self.p.cuda
        if self.use_cuda:
            self.model = self.model.cuda()
            if self.trainable:
                self.loss = self.loss.cuda()

    def _print_params(self):
        """Formats parameters to print when training."""
        if self.trainable:
            print('Training parameters: ')
        else:
            print('Testing/Eval parameters: ')
        self.p.cuda = self.use_cuda
        param_dict = vars(self.p)
        pretty = lambda x: x.replace('_', ' ').capitalize()
        print('\n'.join('  {} = {}'.format(pretty(k), str(v)) for k, v in param_dict.items()))
        print()

    def save_model(self, epoch, stats, first=False):
        """Saves model to files; can be overwritten at every epoch to save disk space."""

        # Create directory for model checkpoints, if nonexistent
        if first:
            ckpt_dir_name = f'{self.job_name}'  # ex: 01-n2npt-train-bernoulli

            self.ckpt_dir = os.path.normpath(os.path.join(self.p.ckpt_save_path, ckpt_dir_name))
            if not os.path.isdir(self.p.ckpt_save_path):
                os.mkdir(self.p.ckpt_save_path)
            if not os.path.isdir(self.ckpt_dir):
                os.mkdir(self.ckpt_dir)

        # Save checkpoint dictionary
        if self.p.ckpt_overwrite:
            fname_unet = '{}/n2n-{}.pt'.format(self.ckpt_dir, self.p.noise_type)  # ex: 'ckpts/01-n2npt-train-bernoulli/n2n-bernoulli.pt'
        else:
            valid_loss = stats['valid_loss'][epoch]
            fname_unet = '{}/n2n-epoch{}-{:>1.5f}.pt'.format(self.ckpt_dir, epoch + 1, valid_loss)
        if self.p.verbose or self.p.show_progress or (epoch + 1) == self.p.nb_epochs:
            print('Saving checkpoint to: {}\n'.format(fname_unet))
        torch.save(self.model.state_dict(), fname_unet)

        # Save stats to JSON
        fname_dict = '{}/n2n-stats.json'.format(self.ckpt_dir)
        with open(fname_dict, 'w') as fp:
            json.dump(stats, fp, indent=2)

    def load_model(self, ckpt_fname):
        """Loads model from checkpoint file."""

        print('Loading checkpoint from: {}\n'.format(ckpt_fname))
        try:
            if self.use_cuda:
                self.model.load_state_dict(torch.load(ckpt_fname))
            else:
                self.model.load_state_dict(torch.load(ckpt_fname, map_location='cpu'))
        except RuntimeError:
            raise ValueError("There is a mismatch between the UNet and the loaded checkpoint. " +
                             "Try checking the requested number of channels and ensure it matches what the checkpoint was trained on. " +
                             "\n(PNG & JPG => 3 channels, XYZ => 1 or 3)")

    def _on_epoch_end(self, stats, train_loss, epoch, epoch_start, valid_loader):
        """Tracks and saves starts after each epoch."""

        # Evaluate model on validation set
        if self.p.verbose or self.p.show_progress:
            print('\rTesting model on validation set... ', end='')
        epoch_time = time_elapsed_since(epoch_start)[0]
        valid_loss, valid_time, valid_psnr = self.eval(valid_loader)
        if self.p.verbose or self.p.show_progress or (epoch + 1) == self.p.nb_epochs:
            show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_psnr)

        # Decrease learning rate if plateau
        self.scheduler.step(valid_loss)

        # Save checkpoint
        stats['train_loss'].append(train_loss)
        stats['valid_loss'].append(valid_loss)
        stats['valid_psnr'].append(valid_psnr)

        self.save_model(epoch, stats, first=(epoch == 0))

        # Plot stats
        loss_str = f'{self.p.loss.upper()} loss'
        plot_per_epoch(self.ckpt_dir, 'Valid loss', stats['valid_loss'], loss_str)
        plot_per_epoch(self.ckpt_dir, 'Valid PSNR', stats['valid_psnr'], 'PSNR (dB)')

    def test(self, test_loader, show):
        """Evaluates denoiser on test set."""

        self.model.train(False)

        self._print_params()

        source_imgs = []
        denoised_imgs = []
        clean_imgs = []

        # Create directory for denoised images
        save_path = os.path.normpath(os.path.join(self.p.output, f'denoised-{self.job_name}-{self.job_id}'))  # ex: 'results/denoised-01-n2npt-train-bernoulli-193174/'
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        for batch_idx, (source, target) in enumerate(test_loader):

            source = source.double()
            target = target.double()

            source_imgs.append(source)
            clean_imgs.append(target)

            if self.use_cuda:
                source = source.cuda()

            # Denoise
            denoised_img = self.model(source).detach().double()
            denoised_imgs.append(denoised_img)

        # Squeeze tensors
        source_imgs = [t.squeeze(0) for t in source_imgs]
        denoised_imgs = [t.squeeze(0) for t in denoised_imgs]
        clean_imgs = [t.squeeze(0) for t in clean_imgs]

        # Create montage and save images
        print('Saving images and montages to: {}'.format(save_path))
        for i in range(len(source_imgs)):
            img_name = test_loader.dataset.imgs[i]
            create_montage(img_name, self.p.noise_type, save_path, source_imgs[i], denoised_imgs[i], clean_imgs[i], show, montage_only=self.p.montage_only)

    def eval(self, valid_loader):
        """Evaluates denoiser on validation set."""

        self.model.train(False)

        valid_start = datetime.now()
        loss_meter = AvgMeter()
        psnr_meter = AvgMeter()

        for batch_idx, (source, target) in enumerate(valid_loader):
            if self.use_cuda:
                source = source.cuda()
                target = target.cuda()

            source = source.double()
            target = target.double()

            # Denoise
            source_denoised = self.model(source).double()

            # Update loss
            loss = self.loss(source_denoised, target)
            loss_meter.update(loss.item())

            # Compute PSRN
            # TODO: Find a way to offload to GPU, and deal with uneven batch sizes
            for i in range(self.p.batch_size):
                source_denoised = source_denoised.cpu()
                target = target.cpu()
                psnr_meter.update(psnr(source_denoised[i], target[i]).item())

        valid_loss = loss_meter.avg
        valid_time = time_elapsed_since(valid_start)[0]
        psnr_avg = psnr_meter.avg

        return valid_loss, valid_time, psnr_avg

    def train(self, train_loader, valid_loader):
        """Trains denoiser on training set."""

        self.model.train(True)

        self._print_params()
        num_batches = len(train_loader)
        if num_batches % self.p.report_interval != 0:
            print("Report interval must be a factor of the total number of batches (nbatches = ntrain / batch_size).",
                  "\nnbatches = {} / {} = {} : {} % {} != 0".format((num_batches * self.p.batch_size), self.p.batch_size, num_batches, self.p.report_interval, num_batches),
                  "\nThe report interval has been reset to equal nbatches.\n")
            self.p.report_interval = num_batches  # if the report interval doesn't evenly divide num_batches, reset

        # Dictionaries of tracked stats
        stats = {'noise_type': self.p.noise_type,
                 'noise_param': self.p.noise_param,
                 'train_loss': [],
                 'valid_loss': [],
                 'valid_psnr': []}

        # Main training loop
        train_start = datetime.now()
        for epoch in range(self.p.nb_epochs):
            if self.p.verbose or self.p.show_progress or (epoch + 1) == self.p.nb_epochs:
                print('EPOCH {:d} / {:d}'.format(epoch + 1, self.p.nb_epochs))

            # Some stats trackers
            epoch_start = datetime.now()
            train_loss_meter = AvgMeter()
            loss_meter = AvgMeter()
            time_meter = AvgMeter()

            # Minibatch SGD
            for batch_idx, (source, target) in enumerate(train_loader):
                batch_start = datetime.now()
                if self.p.show_progress:
                    progress_bar(batch_idx, num_batches, self.p.report_interval, loss_meter.val)

                if self.use_cuda:
                    source = source.cuda()
                    target = target.cuda()

                source = source.double()
                target = target.double()

                # Denoise image
                try:
                    source_denoised = self.model(source).double()
                except RuntimeError:
                    raise ValueError("There is a size mismatch between the number of UNet channels and the input data. " +
                                     "Check the requested number of channels. " +
                                     "\n(PNG & JPG => 3 channels, XYZ => 1 or 3)")

                loss = self.loss(source_denoised, target)
                loss_meter.update(loss.item())

                # Zero gradients, perform a backward pass, and update the weights
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # Report/update statistics
                time_meter.update(time_elapsed_since(batch_start)[1])
                if (batch_idx + 1) % self.p.report_interval == 0 and batch_idx:
                    if self.p.verbose or self.p.show_progress:
                        show_on_report(batch_idx, num_batches, loss_meter.avg, time_meter.avg)
                    train_loss_meter.update(loss_meter.avg)
                    loss_meter.reset()
                    time_meter.reset()

            # Epoch end, save and reset tracker
            self._on_epoch_end(stats, train_loss_meter.avg, epoch, epoch_start, valid_loader)
            train_loss_meter.reset()

        train_elapsed = time_elapsed_since(train_start)[0]
        print('Training done! Total elapsed time: {}\n'.format(train_elapsed))
