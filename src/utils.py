#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from datetime import datetime as dt
from typing import Union
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from PIL import Image
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator
from pathos.helpers import cpu_count
from pathos.pools import ProcessPool as Pool
from skimage.metrics import structural_similarity as SSIM

rcParams['font.family'] = 'serif'
# mpl.use('QtAgg')


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

    timedelta = dt.now() - start
    # string = str(timedelta)[:-7]
    string = str(timedelta)
    ms = int(timedelta.total_seconds() * 1000)

    return string, ms, timedelta


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


def trainvalid_metric_plots(ckpt_dir, train_metric, valid_metric, metric_name):
    fig, ax = plt.subplots(dpi=200)
    if train_metric is not None:
        ax.plot(range(1, len(train_metric) + 1), train_metric, label=f'Train {metric_name}')
    ax.plot(range(1, len(valid_metric) + 1), valid_metric, label=f'Valid {metric_name}')
    ax.set(xlabel='Epoch',
           ylabel=f'{metric_name}',
           title=f"{'Train and Valid' if train_metric is not None else 'Valid'} {metric_name}")
    ax.legend(loc='upper right')
    fig.tight_layout()

    plt.savefig(os.path.join(ckpt_dir,
                             f"{'train-valid' if train_metric is not None else 'valid'}-{metric_name.replace(' ', '-').lower()}.png"))
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

    if source_t.is_floating_point():
        source = tvF.to_pil_image(rescale_tensor(source_t, as_image=True))
        denoised = tvF.to_pil_image(rescale_tensor(denoised_t, as_image=True))
        clean = tvF.to_pil_image(rescale_tensor(clean_t, as_image=True))
    else:
        source = tvF.to_pil_image(source_t)
        denoised = tvF.to_pil_image(torch.clamp(denoised_t, 0, 1))
        clean = tvF.to_pil_image(clean_t)
    # torch.clamp() is like clipping, why is it necessary? (like for RGB?)

    # Build image montage
    psnr_vals = [psnr(source_t, clean_t), psnr(denoised_t, clean_t)]
    ssim_vals = [SSIM(np.asarray(source), np.asarray(clean)), SSIM(np.asarray(denoised), np.asarray(clean))]
    titles = ['Input: {:.2f} dB'.format(psnr_vals[0]),
              'Denoised: {:.2f} dB'.format(psnr_vals[1]),
              'Clean target']
    zipped = zip(titles, [source, denoised, clean])
    for j, (title, img) in enumerate(zipped):
        ax[j].imshow(img, cmap='gray')  # cmap for height fields (not normalized)
        ax[j].set_title(title, fontsize=44)
        ax[j].axis('off')
    fig.tight_layout()

    # Open pop up window, if requested
    # if show > 0:
    #     mpl.use('TkAgg')
    # plt.show()

    # Save to files
    fname = os.path.splitext(img_name)[0]
    with open(os.path.join(save_path, 'metrics.csv'), 'a') as f:
        f.write(
            f'{fname},{round(psnr_vals[0].item(), 2)},{round(psnr_vals[1].item(), 2)},{round(ssim_vals[0], 4)},{round(ssim_vals[1], 4)}\n')

    if not montage_only:
        source.save(os.path.join(save_path, f'{fname}-{noise_type}{noise_param if noise_type != "raw" else ""}-noisy.png'))
        denoised.save(os.path.join(save_path, f'{fname}-{noise_type}{noise_param if noise_type != "raw" else ""}-denoised.png'))
        clean.save(os.path.join(save_path, f'{fname}-{noise_type}{noise_param if noise_type != "raw" else ""}-target.png'))

    fig.savefig(os.path.join(save_path, f'{fname}-{noise_type}{noise_param if noise_type != "raw" else ""}-montage.png'), bbox_inches='tight')
    plt.close()


def plot_tensors(corr_im_tensor: torch.Tensor,
                 source: Union[str, torch.Tensor] = None,
                 titles: list = None,
                 save_path: str = '',
                 f_name: str = '',
                 small_fig: bool = False,
                 cmap=mpl.colormaps['gray']):
    if corr_im_tensor.is_floating_point():  # if tensor isn't already in image scale
        im_pil = tvF.to_pil_image(rescale_tensor(corr_im_tensor, as_image=True))
    else:
        im_pil = tvF.to_pil_image(corr_im_tensor.to(torch.uint8))

    fig_height = 4 if small_fig else 8
    plt.rc('axes', titlesize=28, labelsize=18)  # fontsize of the axes title
    plt.rc('ytick', labelsize=16)
    mpl.rcParams['font.family'] = 'serif'

    if source is not None:  # if fpath is given, assume source is different from im_tensor and plot both
        if titles is None:
            titles = ['Corrupted Image', 'Source Image']
        else:
            assert len(titles) == 2, "The number of titles doesn't match the number of images (2)! Check titles input."

        if type(source) is str:
            source_pil, source_tensor = import_spm(source)
            fignum = os.path.basename(source)
        else:
            source_tensor = source
            if f_name != '':
                fignum = os.path.splitext(f_name)[0]
            else:
                fignum = None
            if source.is_floating_point():
                source_pil = tvF.to_pil_image(rescale_tensor(source, as_image=True))
            else:
                source_pil = tvF.to_pil_image(source.to(torch.uint8))

        im_ratio = max(corr_im_tensor.size()[2], source_tensor.size()[2]) / max(corr_im_tensor.size()[1],
                                                                                corr_im_tensor.size()[1])
        fig, ax = plt.subplots(1, 2, figsize=(im_ratio * 2 * fig_height + 1, fig_height), num=fignum)

        norm_i = mpl.colors.Normalize(vmin=torch.amin(corr_im_tensor).item(), vmax=torch.amax(corr_im_tensor).item())
        sm_i = plt.cm.ScalarMappable(cmap=cmap, norm=norm_i)
        ax[0].imshow(im_pil, cmap=cmap)
        ax[0].set(title=titles[0], xticks=[], yticks=[])
        divider = make_axes_locatable(ax[0])
        cax0 = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(sm_i, cax=cax0)

        norm_s = mpl.colors.Normalize(vmin=torch.amin(source_tensor).item(), vmax=torch.amax(source_tensor).item())
        sm_s = plt.cm.ScalarMappable(cmap=cmap, norm=norm_s)
        ax[1].imshow(source_pil, cmap=cmap)
        ax[1].set(title=titles[1], xticks=[], yticks=[])
        divider = make_axes_locatable(ax[1])
        cax1 = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(sm_s, cax=cax1, label='Z Height or Pixel value')

        plt.subplots_adjust(left=0.01,
                            bottom=0.01,
                            right=0.94,
                            top=0.96,
                            wspace=0.1,
                            hspace=0.0)

        if save_path != '':
            if type(source) is str:
                save_name = os.path.splitext(os.path.basename(source))[
                                0] + f'_montage_{dt.today().strftime("%H%M")}.png'
            elif f_name != '':
                save_name = os.path.splitext(f_name)[0] + f'_montage_{dt.today().strftime("%H%M")}.png'
            else:
                save_name = f'corr_montage_{dt.today().strftime("%H%M%s")}.png'
            # save_path = f'./corrupted_{dt.today().strftime("%Y%m%d")}'
            fig.savefig(os.path.join(save_path, save_name), bbox_inches='tight')
            # plt.close()

    else:
        im_ratio = im_pil.width / im_pil.height
        fig, ax = plt.subplots(figsize=(im_ratio * fig_height + 1, fig_height))
        if titles is not None:
            assert len(titles) == 1, "Need one title for one image! Check titles input."
            ax.set_title(titles[0])
        norm = mpl.colors.Normalize(vmin=torch.amin(corr_im_tensor).item(), vmax=torch.amax(corr_im_tensor).item())
        ax.imshow(im_pil, cmap=cmap)
        ax.set(xticks=[], yticks=[])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, ax=ax, label='Z Height or Pixel value')

        fig.tight_layout()
    plt.ion()
    plt.show()


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
    
    im_tensor = rescale_tensor(im_tensor, dtype=torch.float64)  # rescales to [0, 1]

    return os.path.basename(filepath), im_tensor


def adjust_lr(optimizer, epoch, M, learning_rate=(0.0, 0.001)):
    """ Adjusts optimizer learning rate to a sine wave.
    :param optimizer: optimizer object
    :param epoch: current epoch
    :param M: half-period width of the sinusoid
    :param learning_rate: [min, max] learning rate values
    """
    lr_min = learning_rate[0]
    lr_max = learning_rate[1]
    
    lr_new = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / M))
    for group in optimizer.param_groups:
        group['lr'] = lr_new
    return optimizer


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
