#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF

import os
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
from skimage.metrics import structural_similarity as SSIM
from pathos.helpers import cpu_count
from pathos.pools import ProcessPool as Pool

import matplotlib
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

rcParams['font.family'] = 'serif'
matplotlib.use('agg')


def start_pool():
    return Pool(cpu_count())


def stop_pool(pool):
    pool.close()
    pool.join()
    pool.terminate()


def clear_line():
    """Clears line from any characters."""

    print('\r{}'.format(' ' * 80), end='\r')


def progress_bar(batch_idx, num_batches, report_interval, train_loss):
    """Neat progress bar to track training."""

    dec = int(np.ceil(np.log10(num_batches)))
    bar_size = 27 + dec
    progress = (batch_idx % report_interval) / report_interval
    fill = int(progress * bar_size) + 1
    print('\rBatch {:>{dec}d} [{}{}] Train loss: {:>1.5f}'.format(batch_idx + 1, '=' * fill + '>',
                                                                  ' ' * (bar_size - fill), train_loss,
                                                                  dec=str(dec)), end='')


def time_elapsed_since(start):
    """Computes elapsed time since start."""

    timedelta = datetime.now() - start
    # string = str(timedelta)[:-7]
    string = str(timedelta)
    ms = int(timedelta.total_seconds() * 1000)

    return string, ms, timedelta


def show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_psnr):
    """Formats validation error stats."""

    clear_line()
    print('Train time: {} | Valid time: {} | Valid loss: {:>1.5f} | Avg PSNR: {:.2f} dB'.format(epoch_time[:-4],
                                                                                                valid_time[:-4],
                                                                                                valid_loss, valid_psnr))


def show_on_report(batch_idx, num_batches, loss, elapsed):
    """Formats training stats."""

    clear_line()
    dec = int(np.ceil(np.log10(num_batches)))
    print('Batch {:>{dec}d} / {:d} | Avg loss: {:>1.5f} | Avg train time / batch: {:d} ms'.format(batch_idx + 1,
                                                                                                  num_batches, loss,
                                                                                                  int(elapsed),
                                                                                                  dec=dec))


def plot_per_epoch(ckpt_dir, title, measurements, y_label):
    """Plots stats (train/valid loss, avg PSNR, etc.)."""

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, len(measurements) + 1), measurements)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Epoch')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.tight_layout()

    fname = '{}.png'.format(title.replace(' ', '-').lower())
    plot_fname = os.path.join(ckpt_dir, fname)
    plt.savefig(plot_fname, dpi=200)
    plt.close()


def trainvalid_loss_plots(ckpt_dir, loss_str, train_loss, valid_loss):
    fig, ax = plt.subplots(dpi=200)
    ax.plot(range(1, len(train_loss) + 1), train_loss, label='Train Loss')
    ax.plot(range(1, len(valid_loss) + 1), valid_loss, label='Valid Loss')
    ax.set(xlabel='Epoch', ylabel=f'{loss_str}, Loss', title="Train and Valid Loss")
    ax.legend('upper right')
    fig.tight_layout()

    plt.savefig(os.path.join(ckpt_dir, f'train-valid-loss.png'))
    plt.close()


def reinhard_tonemap(tensor):
    """Reinhard et al. (2002) tone mapping."""

    tensor[tensor < 0] = 0
    return torch.pow(tensor / (1 + tensor), 1 / 2.2)


def psnr(input, target):
    """Computes peak signal-to-noise ratio."""

    return 10 * torch.log10(1 / F.mse_loss(input, target))


def create_montage(img_name, noise_type, noise_param, save_path, source_t, denoised_t, clean_t, show,
                   montage_only=False):
    """Creates montage for easy comparison."""

    fig, ax = plt.subplots(1, 3, figsize=(29, 10), dpi=200)
    fig.canvas.manager.set_window_title(img_name.capitalize()[:-4])

    # Bring tensors to CPU
    # source_t = source_t.cpu().narrow(0, 0, 3)     # commenting this out bc it only works for 3-channel images and may not be necessary anyway
    source_t = source_t.cpu()
    denoised_t = denoised_t.cpu()
    clean_t = clean_t.cpu()

    source = tvF.to_pil_image(source_t)
    denoised = tvF.to_pil_image(torch.clamp(denoised_t, 0, 1))
    clean = tvF.to_pil_image(clean_t)
    # torch.clamp() is like clipping, why is it necessary? (like for RGB?)

    # Build image montage
    psnr_vals = [psnr(source_t, clean_t), psnr(denoised_t, clean_t)]
    ssim_vals = [SSIM(np.asarray(source), np.asarray(clean)), SSIM(np.asarray(denoised), np.asarray(clean))]
    titles = ['Input: {:.2f} dB'.format(psnr_vals[0]),
              'Denoised: {:.2f} dB'.format(psnr_vals[1]),
              'Ground truth']
    zipped = zip(titles, [source, denoised, clean])
    for j, (title, img) in enumerate(zipped):
        ax[j].imshow(img, cmap='gray')  # cmap for height fields (not normalized)
        ax[j].set_title(title, fontsize=44)
        ax[j].axis('off')
    fig.tight_layout()

    # Open pop up window, if requested
    # if show > 0:
    #     matplotlib.use('TkAgg')
    # plt.show()

    # Save to files
    fname = os.path.splitext(img_name)[0]
    f = open(os.path.join(save_path, 'metrics.csv'), 'a')
    # f.write("{:.2f},{:.2f}\n".format(psnr_vals[0], psnr_vals[1]))
    f.write(f'{fname},{round(psnr_vals[0], 2)},{round(psnr_vals[1], 2)},{round(ssim_vals[0], 2)},{round(ssim_vals[1], 2)}')
    f.close()

    if not montage_only:
        source.save(os.path.join(save_path, f'{fname}-{noise_type}{noise_param}-noisy.png'))
        denoised.save(os.path.join(save_path, f'{fname}-{noise_type}{noise_param}-denoised.png'))
        clean.save(os.path.join(save_path, f'{fname}-{noise_type}{noise_param}-target.png'))

    fig.savefig(os.path.join(save_path, f'{fname}-{noise_type}{noise_param}-montage.png'), bbox_inches='tight')
    plt.close()


def rescale_tensor(tensor, as_image=False, bounds=(0, 1), dtype=None):
    """ Scales an input tensor to between 0 and 1 or given maximum.

    :param tensor: torch tensor input
    :param as_image: changes scaling range to 0-255 and converts to dtype=uint8
    :param bounds: sets custom maximum value
    :param dtype: tensor dtype to use for output
    :return: rescaled tensor with desired dtype
    """

    numer = tensor - tensor.min().item()
    denom = (tensor.max().item() - tensor.min().item())

    if as_image:
        rescaled = ((255 * numer) / denom).to(torch.uint8)
    else:
        if dtype is None:
            dtype = tensor.dtype
        rescaled = (bounds[0] + ((bounds[1] - bounds[0]) * numer) / denom).to(dtype)

    return rescaled


def import_spm(filepath):
    picture = ['.png', '.jpeg', '.jpg']
    xyz = ['.xyz', '.csv', '.txt']
    ext = os.path.splitext(filepath)[-1].lower()

    if ext not in (picture + xyz):
        raise TypeError(f'File must be in the following supported formats: {picture + xyz}')

    if ext.lower() in picture:
        with Image.open(filepath).convert("L") as im_pil:
            im_pil.load()
        im_tensor = tvF.pil_to_tensor(im_pil)

    else:  # ext.lower() in xyz:
        df = pd.read_csv(filepath, header=None, delimiter='\t', index_col=False)
        P = int(np.sqrt(len(df)))  # get img dimension
        try:
            im_tensor = torch.tensor(df[df.columns[-1]].values.reshape((P, -1))).unsqueeze(
                dim=0)  # reshape using img dim
        except ValueError:  # just in case above doesn't work, find highest power of 2 to reshape
            P = int(pow(2, int(np.log2(P))))
            im_tensor = torch.tensor(df[df.columns[-1]].values.reshape((P, -1))).unsqueeze(dim=0)

        # im_pil = tvF.to_pil_image(rescale_tensor(im_tensor, as_image=True), mode="L")

    return os.path.basename(filepath), im_tensor


class AvgMeter(object):
    """Computes and stores the average and current value.
    Useful for tracking averages such as elapsed times, minibatch losses, etc.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0.
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
