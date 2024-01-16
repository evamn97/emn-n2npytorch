#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
from datetime import timedelta

import torch.nn as nn
from torch.optim import Adam, lr_scheduler

from unet import UNet
from utils import *


class Noise2Noise(object):
    """Implementation of Noise2Noise from Lehtinen et al. (2018)."""

    def __init__(self, params, trainable):
        """Initializes model."""
        self.n2n_start = datetime.now()
        print(f'N2N start time:     {self.n2n_start.strftime("%H:%M:%S.%f")[:-4]}')
        self.epoch_times = []  # added 6/24/22
        self.p = params
        self.trainable = trainable
        self._compile()  # creates unet
        self.ckpt_dir = ''

        # try to sync montage output with SLURM output filenames by getting job ID and name - emn 05/19/22
        if "jobid" in os.environ:
            self.job_id = os.environ["jobid"]  # ex: 193174
        else:
            self.job_id = f'{datetime.now():%m%d%H%M}'  # ex: 05311336
        if "jobname" in os.environ and "idv" not in os.environ["jobname"]:
            self.job_name = os.environ["jobname"]  # ex: tgx-n2npt-train-bernoulli
        elif "filename" in os.environ:
            self.job_name = f'{os.environ["filename"].replace("-imgrec", "")}{("redux" + self.p.redux) if self.p.redux > 0 else ""}-{self.p.noise_type}{"clean" if (trainable and self.p.clean_targets) else ""}{self.p.noise_param if self.p.noise_type != "raw" else ""}{self.p.loss}'  # ex: tinyimagenetredux0.9-bernoulli0.5l1
        else:
            self.job_name = f'{self.p.noise_type}{"-clean" if (trainable and self.p.clean_targets) else ""}{self.job_id}'

        # debugging
        print(f'n2n initialized in   {str(datetime.now() - self.n2n_start)[:-4]} from n2n start')
        sys.stdout.flush()

    def _compile(self):
        """Compiles model (architecture, loss function, optimizers, etc.)."""

        # Model
        self.model = UNet(in_channels=self.p.channels, out_channels=self.p.channels)  # .double()  # changed this from default in_channels=3 6/13/22 emn

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
        sys.stdout.flush()

    def save_model(self, epoch, stats, first=False):
        """Saves model to files; can be overwritten at every epoch to save disk space."""

        # Create directory for model checkpoints, if nonexistent
        if first:
            if self.p.load_ckpt and os.path.isfile(self.p.load_ckpt):  # indicate previous ckpt in ckpt_dir_name
                if "retrain" in self.p.load_ckpt:
                    # self.ckpt_dir = os.path.basename(os.path.dirname(self.p.load_ckpt)) + "-{}{}".format(self.p.noise_param, self.p.loss)
                    self.ckpt_dir = f'{os.path.basename(os.path.dirname(self.p.load_ckpt))}{("redux" + self.p.redux) if self.p.redux > 0 else ""}-{self.p.noise_param}{self.p.loss}'
                else:
                    # self.ckpt_dir = os.path.basename(os.path.dirname(self.p.load_ckpt)) + "-retrain-{}{}".format(self.p.noise_param, self.p.loss)  # ex: hs20mg-bernoulli-retrain-0.4l1
                    self.ckpt_dir = f'{os.path.basename(os.path.dirname(self.p.load_ckpt))}-retrain{("redux" + self.p.redux) if self.p.redux > 0 else ""}-{self.p.noise_param}{self.p.loss}'
            else:
                # self.ckpt_dir = self.p.ckpt_save_path
                self.ckpt_dir = os.path.join(self.p.ckpt_save_path, f'{self.job_name}')

            if not os.path.isdir(self.ckpt_dir):
                os.makedirs(self.ckpt_dir)

        # Save checkpoint dictionary
        if self.p.ckpt_overwrite:
            ckpt_filename = f'{self.job_name}{self.p.noise_param}{self.p.loss}-e{self.p.nb_epochs}'  # ex: hs20mg-bernoulli0.4l1-e100
            fname_unet = '{}/{}.pt'.format(self.ckpt_dir, ckpt_filename)  # changed 12/18/2022
            # fname_unet = '{}/{}.pt'.format(self.ckpt_dir, os.path.basename(self.ckpt_dir))  # changed 7/5/2022
        else:
            valid_loss = stats['valid_loss'][epoch]
            fname_unet = '{}/n2n-epoch{}-{:>1.5f}.pt'.format(self.ckpt_dir, epoch + 1, valid_loss)
        if self.p.verbose or (epoch + 1) == self.p.nb_epochs:
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
                             "Try checking the requested number of channels and ensure it matches what the checkpoint was pretrained on. ")

    def _on_epoch_end(self, stats, train_loss, epoch, epoch_start, valid_loader):
        """Tracks and saves starts after each epoch."""

        # Evaluate model on validation set
        if self.p.verbose:
            print('\rTesting model on validation set... ', end='')
        epoch_time, _, epoch_time_td = time_elapsed_since(epoch_start)
        valid_loss, valid_time, valid_time_td, valid_psnr = self.eval(valid_loader)

        self.epoch_times.append(epoch_time_td + valid_time_td)

        if self.p.verbose or (epoch + 1) == self.p.nb_epochs:
            show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_psnr)
            est_remain = (sum(self.epoch_times, timedelta(0)) / len(self.epoch_times)) * (self.p.nb_epochs - (epoch + 1))
            print(f'Estimated time remaining: {str(est_remain)[:-7]}')
            sys.stdout.flush()  # force print to out file

        # Decrease learning rate if plateau
        self.scheduler.step(valid_loss)

        # Save checkpoint
        stats['train_loss'].append(train_loss)
        stats['valid_loss'].append(valid_loss)
        stats['valid_psnr'].append(valid_psnr)

        self.save_model(epoch, stats, first=(epoch == 0))

        # Plot stats
        loss_str = f'{self.p.loss.upper()} loss'
        plot_per_epoch(self.ckpt_dir, ('Valid ' + loss_str), stats['valid_loss'], loss_str)
        plot_per_epoch(self.ckpt_dir, 'Valid PSNR', stats['valid_psnr'], 'PSNR (dB)')

        ratio = list(np.divide(stats['valid_psnr'], stats['valid_loss']))  # PSNR/Loss
        plot_per_epoch(self.ckpt_dir, ('Valid PSNR over ' + loss_str), ratio, '')

    def test(self, test_loader, show):
        """Evaluates denoiser on test set."""

        self.model.train(False)

        self._print_params()

        source_imgs = []
        denoised_imgs = []
        clean_imgs = []

        # Create directory for denoised images
        if not os.path.isdir(self.p.output):
            os.mkdir(self.p.output)
        subfolder = f'{self.job_name.replace("imgrec", "denoised").replace("-test", "")}-{self.job_id}'
        save_path = os.path.normpath(os.path.join(self.p.output, subfolder))  # ex: 'results/hs20mg-bernoulli0.4l2-193174/'
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        for batch_idx, (source, target) in enumerate(test_loader):

            # source = source.double()
            # target = target.double()

            source_imgs.append(source)
            clean_imgs.append(target)

            if self.use_cuda:
                source = source.cuda()

            # Denoise
            denoised_img = self.model(source).detach()  # .double()
            denoised_imgs.append(denoised_img)

        # Squeeze tensors
        source_imgs = [t.squeeze(0) for t in source_imgs]
        denoised_imgs = [t.squeeze(0) for t in denoised_imgs]
        clean_imgs = [t.squeeze(0) for t in clean_imgs]

        # Create montage and save images
        print('Saving images and montages to: {}'.format(save_path))
        if not os.path.isfile(os.path.join(save_path, 'metrics.csv')):
            f = open(os.path.join(save_path, 'metrics.csv'), 'w')  # create a text file to save psnr values to
            f.write("file,psnr_in,psnr_out,ssim_in,ssim_out\n")
            f.close()

        for i in range(len(source_imgs)):
            img_name = test_loader.dataset.img_fnames[i]
            create_montage(img_name, self.p.noise_type, self.p.noise_param, save_path,
                           source_imgs[i], denoised_imgs[i], clean_imgs[i],
                           show, montage_only=self.p.montage_only)

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

            # source = source.double()
            # target = target.double()

            # Denoise
            source_denoised = self.model(source) # .double()

            # Update loss
            loss = self.loss(source_denoised, target)
            loss_meter.update(loss.item())

            # Compute PSNR
            # TODO: Find a way to offload to GPU
            for i in range(self.p.batch_size):
                source_denoised = source_denoised.cpu()
                target = target.cpu()
                try:
                    psnr_meter.update(psnr(source_denoised[i], target[i]).item())
                except IndexError:
                    # this will trigger when batch size causes uneven division of data (with remainder)
                    # so final batch is smaller than given batch size and the loop goes out of bounds

                    break

        valid_loss = loss_meter.avg
        valid_time, _, valid_time_td = time_elapsed_since(valid_start)
        psnr_avg = psnr_meter.avg

        return valid_loss, valid_time, valid_time_td, psnr_avg

    def train(self, train_loader, valid_loader):
        """Trains denoiser on training set."""

        self.model.train(True)

        self._print_params()
        num_batches = len(train_loader)
        if num_batches % self.p.report_interval != 0:
            print("Report interval must be a factor of the total number of batches (nbatches = ntrain / batch_size).",
                  "\nnbatches = {}/{} = {}, report_interval = {}:   {} % {} != 0".format((num_batches * self.p.batch_size), self.p.batch_size, num_batches, self.p.report_interval, self.p.report_interval, num_batches),
                  "\nThe report interval has been reset to equal nbatches ({}).\n".format(num_batches))
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
            if self.p.verbose or (epoch + 1) == self.p.nb_epochs:
                print('EPOCH {:d} / {:d}'.format(epoch + 1, self.p.nb_epochs))
                sys.stdout.flush()

            # Some stats trackers
            epoch_start = datetime.now()
            train_loss_meter = AvgMeter()
            loss_meter = AvgMeter()
            time_meter = AvgMeter()

            # Minibatch SGD
            loop_start = datetime.now()
            train_iter = enumerate(train_loader)
            for batch_idx, (source, target) in train_iter:  # enumerate(train_loader):
                batch_start = datetime.now()
                if self.p.show_progress:
                    progress_bar(batch_idx, num_batches, self.p.report_interval, loss_meter.val)

                if self.use_cuda:
                    source = source.cuda()
                    target = target.cuda()

                # source = source.double()
                # target = target.double()

                self.optim.zero_grad()  # zero gradient before step

                # Denoise image
                try:
                    source_denoised = self.model(source)  # .double()
                except RuntimeError:
                    raise ValueError(f"This error can be thrown if the input data dimensions don't match the defined input channels of the model"
                                     f"\t source shape (where this error is thrown) should have len = 4 (e.g., [4, 1, 128, 128]), and dim 1 should match in_channels: source.shape = {source.shape}, in_channels = {self.p.channels}",
                                     f"\nOr it can be caused by a dtype mismatch between the bias and input.",
                                     f"\tBias type is usually float32, input dtype = {source.dtype}")

                loss = self.loss(source_denoised, target)
                loss_meter.update(loss.item())

                loss.backward()     # takes a bit longer?
                self.optim.step()

                # Report/update statistics
                time_meter.update(time_elapsed_since(batch_start)[1])
                if (batch_idx + 1) % self.p.report_interval == 0 and batch_idx:
                    if self.p.verbose:
                        if self.p.show_progress:
                            print("")
                        show_on_report(batch_idx, num_batches, loss_meter.avg, time_meter.avg)
                        sys.stdout.flush()
                    train_loss_meter.update(loss_meter.avg)
                    loss_meter.reset()
                    time_meter.reset()

                print(f'batch time: {datetime.now() - batch_start} | loop time: {datetime.now() - loop_start}')
                loop_start = datetime.now()

            if self.p.verbose or (epoch + 1) == self.p.nb_epochs:
                print('Epoch {} finished at {} from n2n start.'.format((epoch + 1), str((datetime.now() - self.n2n_start))[:-4]))
                sys.stdout.flush()

            # Epoch end, save and reset tracker
            self._on_epoch_end(stats, train_loss_meter.avg, epoch, epoch_start, valid_loader)
            train_loss_meter.reset()

        train_elapsed = time_elapsed_since(train_start)[0]
        trainvalid_loss_plots(self.ckpt_dir, self.p.loss.upper(), stats['train_loss'], stats['valid_loss'])
        print('Training done! Total elapsed time: {}'.format(str(train_elapsed)[:-3]))
        print('Average training time per epoch:   {}\n'.format(str(sum(self.epoch_times, timedelta(0)) / len(self.epoch_times))[:-3]))
