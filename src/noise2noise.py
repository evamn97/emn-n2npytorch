#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
from datetime import timedelta

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts as CosAnn
from torchmetrics import MeanAbsoluteError as MAE
from metrics import MSE_SSIM_Loss, LPIPS_Loss, PSNR_Meter, SSIM_Meter
from tqdm import tqdm

from unet import UNet
from utils import *


class Noise2Noise(object):
    """Implementation of Noise2Noise from Lehtinen et al. (2018)."""

    def __init__(self, params, trainable):
        """Initializes model."""
        self.n2n_start = dt.now()
        self.epoch_times = []  # added 6/24/22
        self.p = params
        self.trainable = trainable
        self._compile()  # creates unet
        self.ckpt_dir = ''

        # try to sync montage output with SLURM output filenames by getting job ID and name - emn 05/19/22
        self.job_id = f'{dt.now():%m%d%H%M}'  # ex: 05311336
        self.job_name = f'{self.p.noise_type}{"-clean" if (trainable and self.p.clean_targets) else ""}{self.job_id}'

        # debugging
        # print(f'n2n initialized in:  {str(dt.now() - self.n2n_start)[:-4]}')
        # sys.stdout.flush()

    def _compile(self):
        """Compiles model (architecture, loss function, optimizers, etc.)."""

        # Model
        self.model = UNet(in_channels=self.p.channels,
                          out_channels=self.p.channels)  # .double()  # changed this from default in_channels=3 6/13/22 emn

        # Set optimizer and loss, if in training mode
        if self.trainable:
            self.optim = Adam(self.model.parameters(),
                              lr=self.p.learning_params[1],
                              betas=self.p.adam[:2],
                              eps=self.p.adam[2])

            # Learning rate adjustment
            self.scheduler = CosAnn(self.optim,
                                    T_0=int(self.p.learning_params[2]),
                                    T_mult=int(self.p.learning_params[3]),
                                    eta_min=self.p.learning_params[0])

            # Loss function
            if self.p.loss == 'l2':
                # self.loss = nn.MSELoss()
                # self.loss = MSE()
                self.loss = MSE_SSIM_Loss(data_range=(0.0, 1.0))
            else:
                # self.loss = nn.L1Loss()
                self.loss = MAE()
                
        # self.psnr = PSNR(data_range=(0.0, 1.0))
        # self.ssim = SSIM(data_range=(0.0, 1.0))
        self.psnr = PSNR_Meter(data_range=(0.0, 1.0))
        self.ssim = SSIM_Meter(data_range=(0.0, 1.0))

        # CUDA support
        self.use_cuda = torch.cuda.is_available() and self.p.cuda
        if self.use_cuda:
            self.model = self.model.cuda()
            if self.trainable:
                self.loss = self.loss.cuda()
            self.psnr = self.psnr.cuda()
            self.ssim = self.ssim.cuda()

    def _print_params(self):
        """Formats parameters to print when training."""
        if self.trainable:
            print('\nTraining parameters: ')
        else:
            print('\nTesting/Eval parameters: ')
        self.p.cuda = self.use_cuda
        param_dict = vars(self.p)
        pretty = lambda x: x.replace('_', ' ').capitalize()
        print('\n'.join('  {} = {}'.format(pretty(k), str(v)) for k, v in param_dict.items()))
        print()
        sys.stdout.flush()

    def show_on_report(self, batch_idx, num_batches, loss, elapsed):
        """Formats training stats."""
        print('\r', end='')
        out_str = f'Batch {batch_idx + 1}/{num_batches} | Avg Loss: {loss} | Avg time per batch: {int(elapsed)}'
        if self.p.lr_scheduler:
            out_str += f' | LR: {self.scheduler.get_last_lr()[0]}'

        print(out_str)

    def save_model(self, epoch, stats, first=False):
        """Saves model to files; can be overwritten at every epoch to save disk space."""

        # Create directory for model checkpoints, if nonexistent
        if first:
            if self.p.load_ckpt and os.path.isfile(self.p.load_ckpt):  # indicate previous ckpt in ckpt_dir_name
                if "retrain" in self.p.load_ckpt:
                    ckpt_subdir = f'{os.path.basename(os.path.dirname(self.p.load_ckpt))}_{("redux" + str(self.p.redux)) if self.p.redux > 0 else ""}-{self.p.noise_param if self.p.noise_type != "raw" else ""}{self.p.loss}'
                else:
                    ckpt_subdir = f'{os.path.basename(os.path.dirname(self.p.load_ckpt))}_retrain{("redux" + str(self.p.redux)) if self.p.redux > 0 else ""}-{self.p.noise_param if self.p.noise_type != "raw" else ""}{self.p.loss}'
            else:
                ckpt_subdir = f'{self.job_name}'

            self.ckpt_dir = os.path.normpath(os.path.join(self.p.ckpt_save_path, ckpt_subdir))

            if os.path.isdir(self.ckpt_dir):
                idx = sum(ckpt_subdir in dirname for dirname in os.listdir(self.p.ckpt_save_path) if
                          os.path.isdir(os.path.join(self.p.ckpt_save_path, dirname)))
                self.ckpt_dir += f'-{idx}'

            os.makedirs(self.ckpt_dir)

        # Save checkpoint dictionary
        if self.p.ckpt_overwrite:
            ckpt_filename = f'{self.job_name}{self.p.noise_param}{self.p.loss}-e{self.p.nb_epochs}'  # ex: hs20mg-bernoulli0.4l1-e100
            fname_unet = '{}/{}.pt'.format(self.ckpt_dir, ckpt_filename)
        else:
            valid_loss = stats['valid_loss'][epoch]
            fname_unet = '{}/train-epoch{}-{:>1.5f}.pt'.format(self.ckpt_dir, epoch + 1, valid_loss)
        if self.p.verbose or (epoch + 1) == self.p.nb_epochs:
            print('Saving checkpoint to: {}'.format(fname_unet))
        torch.save(self.model.state_dict(), fname_unet)

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
                             "Try checking the requested number of channels and ensure it matches what the checkpoint was (pre-)trained on. ")

    def _on_epoch_end(self, stats, train_loss, epoch, epoch_start, valid_loader):
        """Tracks and saves starts after each epoch."""

        # Evaluate model on validation set
        valid_loss, valid_time, valid_time_td, valid_psnr, valid_ssim = self.eval(valid_loader)

        # # Decrease learning rate if plateau (ReduceLRonPlateau scheduler)
        # self.scheduler.step(valid_loss)

        # Save stats
        stats['train_loss'].append(train_loss)
        stats['valid_loss'].append(valid_loss)
        stats['valid_psnr'].append(valid_psnr)
        stats['valid_ssim'].append(valid_ssim)

        # Save checkpoint
        if epoch == 0 or (epoch + 1) % self.p.ckpt_save_every == 0 or (epoch + 1) == self.p.nb_epochs:
            self.save_model(epoch, stats, first=(epoch == 0))

        # save stat plots
        if epoch > 0:
            trainvalid_metric_plots(self.ckpt_dir, None, stats['valid_psnr'], 'PSNR')
            trainvalid_metric_plots(self.ckpt_dir, None, stats['valid_ssim'], 'SSIM')
            trainvalid_metric_plots(self.ckpt_dir, stats['train_loss'], stats['valid_loss'],
                                    f'{self.p.loss.upper()} Loss')

        # save stats to json file
        fname_dict = '{}/n2n-stats.json'.format(self.ckpt_dir)
        with open(fname_dict, 'w') as fp:
            json.dump(stats, fp, indent=2)

        epoch_time, _, epoch_time_td = time_elapsed_since(epoch_start)
        self.epoch_times.append(epoch_time_td)

        if self.p.verbose:
            print(
                'Epoch time: {} | Valid loss: {:>1.5f} | Valid PSNR: {:.2f} dB | Valid SSIM: {:.2f}'.format(epoch_time[:-4],
                                                                                                        valid_loss,
                                                                                                        valid_psnr,
                                                                                                        valid_ssim))
            est_remain = (sum(self.epoch_times, timedelta(0)) / len(self.epoch_times)) * (
                    self.p.nb_epochs - (epoch + 1))
            print(
                f'Current fit runtime:  {str(dt.now() - self._train_start)[:-4]}\nEst. time remaining:  {str(est_remain)[:-4]}')

            sys.stdout.flush()  # force print to out file

    def test(self, test_loader, show):
        """Evaluates denoiser on test set."""

        self.model.train(False)

        self._print_params()

        source_imgs = []
        denoised_imgs = []
        clean_imgs = []

        # Create directory for denoised images
        subfolder = f'{self.job_name.replace("imgrec", "denoised").replace("-test", "")}-{self.job_id}'
        save_path = os.path.normpath(
            os.path.join(self.p.output, subfolder))  # ex: 'results/hs20mg-bernoulli0.4l2-193174/'
        if os.path.isdir(save_path):
            idx = sum(subfolder in dirname for dirname in os.listdir(self.p.output) if
                      os.path.isdir(os.path.join(self.p.output, dirname)))
            save_path += f'-{idx}'
        os.makedirs(save_path)

        for batch_idx, (source, target) in enumerate(test_loader):

            source_imgs.append(source)
            clean_imgs.append(target)

            if self.use_cuda:
                source = source.cuda()

            # Denoise
            denoised_img = self.model(source).detach()
            denoised_imgs.append(denoised_img)

        # Squeeze tensors
        source_imgs = [t.squeeze(0) for t in source_imgs]
        denoised_imgs = [t.squeeze(0) for t in denoised_imgs]
        clean_imgs = [t.squeeze(0) for t in clean_imgs]

        # Create montage and save images
        print('Saving results to: {}\n'.format(save_path))

        with open(os.path.join(save_path, 'metrics.csv'), 'w') as f:  # create a text file to save psnr values to
            f.write("file,psnr_in,psnr_out,ssim_in,ssim_out\n")

        if self.p.verbose:
            test_iter = tqdm(range(len(source_imgs)), desc='Saving results images', unit='img')
        else:
            test_iter = range(len(source_imgs))
        for i in test_iter:
            img_name = test_loader.dataset.img_fnames[i]
            create_montage(img_name, self.p.noise_type, self.p.noise_param, save_path,
                           source_imgs[i], denoised_imgs[i], clean_imgs[i],
                           show, montage_only=self.p.montage_only)

    def eval(self, valid_loader):
        """Evaluates denoiser on validation set."""

        self.model.train(False)

        valid_start = dt.now()

        if self.p.verbose:
            print('\rTesting model on validation set... ', end='')

        for batch_idx, (source, target) in enumerate(valid_loader):
            if self.use_cuda:
                source = source.cuda()
                target = target.cuda()

            # Denoise
            source_denoised = self.model(source)

            # Calculate metrics
            self.psnr.update(source_denoised, target)
            self.ssim.update(source_denoised, target)

            # Update loss
            self.loss.update(source_denoised, target)

        valid_loss = self.loss.compute().item()
        valid_time, _, valid_time_td = time_elapsed_since(valid_start)
        psnr_avg = self.psnr.compute().item()
        ssim_avg = self.ssim.compute().item()

        self.loss.reset()
        self.psnr.reset()
        self.ssim.reset()

        clear_line()  # clears the "testing on validation.." line

        return valid_loss, valid_time, valid_time_td, psnr_avg, ssim_avg

    def train(self, train_loader, valid_loader):
        """Trains denoiser on training set."""

        self._train_start = dt.now()

        self._print_params()
        num_batches = len(train_loader)
        report_interval = int(num_batches / self.p.report_per_epoch) if self.p.report_per_epoch < num_batches else 1

        # Dictionaries of tracked stats
        stats = {'noise_type': self.p.noise_type,
                 'noise_param': self.p.noise_param,
                 'train_loss': [],
                 'train_psnr': [],
                 'train_ssim': [],
                 'valid_loss': [],
                 'valid_psnr': [],
                 'valid_ssim': []}

        # Main training loop
        for epoch in range(self.p.nb_epochs):
            self.model.train(True)
            if self.p.verbose or (epoch + 1) == self.p.nb_epochs:
                print('\nEPOCH {:d} / {:d}'.format(epoch + 1, self.p.nb_epochs))
                sys.stdout.flush()

            # Some stats trackers
            epoch_start = dt.now()
            time_meter = AvgMeter()

            # Minibatch SGD
            pbar = tqdm(total=len(train_loader), desc=f'Train Epoch {epoch + 1}', unit='batch')
            for batch_idx, (source, target) in enumerate(train_loader):
                batch_start = dt.now()

                if self.use_cuda:
                    source = source.cuda()
                    target = target.cuda()

                self.optim.zero_grad()  # zero gradient before step

                # Denoise image
                try:
                    source_denoised = self.model(source)
                except RuntimeError:
                    raise ValueError(
                        f"This error can be thrown if the input data dimensions don't match the defined input channels of the model"
                        f"\t source shape (where this error is thrown) should have len = 4 (e.g., [4, 1, 128, 128]), and dim 1 should match in_channels: source.shape = {source.shape}, in_channels = {self.p.channels}",
                        f"\nOr it can be caused by a dtype mismatch between the bias and input.",
                        f"\tBias type is usually float32, input dtype = {source.dtype}")

                # Calculate ssims and update loss
                self.psnr.update(source_denoised, target)
                self.ssim.update(source_denoised, target)
                loss = self.loss(source_denoised, target)

                # Update loss and step optimizer
                loss.backward()
                self.optim.step()
                self.scheduler.step(epoch + batch_idx / num_batches)

                pbar.update()
                pbar.set_postfix({'lr': round(self.scheduler.get_last_lr()[0], 6),
                                  'loss': self.loss.compute().item(),
                                  'psnr': self.psnr.compute().item(),
                                  'ssim': self.ssim.compute().item()})

                # Report/update statistics
                time_meter.update(time_elapsed_since(batch_start)[1])
                time_meter.reset()

            # Epoch end, save and reset trackers
            # self.scheduler.step()
            pbar.close()
            stats['train_psnr'].append(self.psnr.compute().item())
            stats['train_ssim'].append(self.ssim.compute().item())
            epoch_loss = self.loss.compute().item()
            self.loss.reset()
            self.psnr.reset()
            self.ssim.reset()
            self._on_epoch_end(stats, epoch_loss, epoch, epoch_start, valid_loader)

        train_elapsed = time_elapsed_since(self._train_start)[0]

        print('\nTraining done! Total elapsed time: {}'.format(str(train_elapsed)[:-3]))
        print('Average training time per epoch:   {}\n'.format(
            str(sum(self.epoch_times, timedelta(0)) / len(self.epoch_times))[:-3]))
