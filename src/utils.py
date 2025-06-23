#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from datetime import datetime as dt
from typing import Union, Literal, Optional
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
import torchvision.transforms as trf
from PIL import Image
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator
from pathos.helpers import cpu_count
from pathos.pools import ProcessPool as Pool
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_F
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr_F
# from skimage.metrics import structural_similarity as SSIM

rcParams['font.family'] = 'serif'
# mpl.use('QtAgg')

PICTURE_TYPES = ['.png', '.jpg', '.jpeg']
XYZ_TYPES = ['.xyz', '.txt', '.csv']
SUPPORTED = PICTURE_TYPES + XYZ_TYPES + ['.pt']


def clear_line():
    """Clears line from any characters."""

    print('\r{}'.format(' ' * 80), end='\r')


def progress_bar(batch_idx, num_batches, report_interval, loss, time):
    """Neat progress bar to track training."""

    dec = int(np.ceil(np.log10(num_batches)))
    bar_size = 27 + dec
    # progress = (batch_idx % report_interval) / report_interval
    progress = batch_idx / num_batches
    fill = int(progress * bar_size) + 1
    print('Batch {:>{dec}d} / {} [{}{}] train loss: {:>1.5f} | time / batch: {:d} ms'.format(batch_idx + 1, num_batches, 
                                                                                             '=' * fill + '>',
                                                                  ' ' * (bar_size - fill), 
                                                                  loss, 
                                                                  int(time),
                                                                  dec=str(dec)), end='\t\t\t\t\r')
    

def time_elapsed_since(start):
    """Computes elapsed time since start."""

    timedelta = dt.now() - start
    # string = str(timedelta)[:-7]
    string = str(timedelta)
    ms = int(timedelta.total_seconds() * 1000)

    return string, ms, timedelta


def show_on_report(batch_idx, num_batches, loss, elapsed):
    """Formats training stats."""

    # clear_line()
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

    valid_metric = np.array(valid_metric)

    if train_metric is not None:
        train_metric = np.array(train_metric)
        if abs(train_metric[0]) >= 100 * abs(train_metric[1]):
            temp_t = train_metric[1:]
            temp_v = valid_metric[1:]
            ylim=(0.98*min(train_metric.min(), valid_metric.min()), 1.02*max(temp_t.max(), temp_v.max()))
            ax.set_ylim(ylim[0], ylim[1])
            ax.text(0, 0.985*ylim[1], 
                    '*Note: epoch 1 value(s) out of bounds', 
                    fontsize='xx-small')
        ax.plot(range(1, len(train_metric) + 1), train_metric, label=f'Train {metric_name}')

    if abs(valid_metric[0]) >= 100 * abs(valid_metric[1]):
        temp_v = valid_metric[1:]
        ylim=(0.98*valid_metric.min(), 1.02*temp_v.max())
        ax.set_ylim(ylim[0], ylim[1])
        ax.text(0, 0.985*ylim[1], 
                '*Note: epoch 1 value(s) out of bounds', 
                fontsize='xx-small')
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


# def PSNR(input, target):
#     """Computes peak signal-to-noise ratio."""

#     return 10 * torch.log10(1 / F.mse_loss(input, target))


def create_montage(img_name, noise_type, noise_param, save_path, source_t, denoised_t, clean_t, show,
                   montage_only=False):
    """Creates montage for easy comparison."""

    fig, ax = plt.subplots(1, 3, figsize=(29, 12.5), dpi=200)
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
    psnr_vals = [psnr_F(source_t, clean_t), psnr_F(denoised_t, clean_t)]
    ssim_vals = [ssim_F(source_t, clean_t), ssim_F(denoised_t, clean_t)]
    specs = [f'PSNR={round(psnr_vals[0].item(), 2)} dB | SSIM={round(ssim_vals[0].item(), 2)}', 
         f'PSNR={round(psnr_vals[1].item(), 2)} dB | SSIM={round(ssim_vals[1].item(), 2)}', 
         '']
    titles = ['Input', 
              'Output', 
              'Clean Target']

    zipped = zip(titles, [source, denoised, clean])
    for j, (title, img) in enumerate(zipped):
        ax[j].imshow(img, cmap='gray')  # cmap for height fields (not normalized)
        ax[j].set_title(title, fontsize=44, pad=15)
        ax[j].set_xlabel(specs[j], fontsize=32, labelpad=10)    
        ax[j].xaxis.set_label_position('top')
        # ax[j].axis('off')
        ax[j].set(xticks=[], yticks=[])

    fig.tight_layout()

    # Open pop up window, if requested
    # if show > 0:
    #     mpl.use('TkAgg')
    #     plt.show()

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


def plot_tensors(*images: Union[Tensor, np.ndarray],
                 titles: Optional[list] = None,
                 save_dir: str = '',
                 f_name: str = '',
                 tag: str = '',
                 cmap=mpl.colormaps['gray'],
                 colorscale: Union[bool, Literal['compare']] = False,
                 show: bool = True,
                 dpi: int = 100,
                 ) -> None:
    """ Plots up to 6 image tensors in a single row. """

    if len(images) == 0:
        raise RuntimeError('"Failure!" - Pre Vizla, TCW S02E12')
    elif len(images) > 6:
        raise RuntimeError(f'{len(images)} is too many')

    fig_height = 4
    plt.rc('axes', titlesize=18, labelsize=13)  # fontsize of the axes title
    plt.rc('ytick', labelsize=12)

    mpl.rcParams['font.family'] = 'serif'

    if f_name != '':
        fignum = os.path.splitext(f_name)[0]
    else:
        fignum = None

    im_ratio = max([im.shape[-1] for im in images]) / max([im.shape[-2] for im in images])
    fig, ax = plt.subplots(1, len(images), figsize=(im_ratio * len(images) * fig_height + 1, fig_height), num=fignum, dpi=dpi)

    if titles is not None:
        assert len(titles) == (len(images)), f"The number of titles doesn't match the number of images ({len(images)})!"

    # only plot one image
    if len(images) == 1:
        if isinstance(images[0], np.ndarray):
            img_tensor = tvF.to_tensor(images[0])
        else:
            img_tensor = images[0].detach()
        img_pil = tvF.to_pil_image(rescale_tensor(img_tensor, as_image=True))
        ax.imshow(img_pil, cmap=cmap)

        if titles is not None:
            ax.set_title(titles[0])
        ax.axis('off')

        if colorscale:
            norm = mpl.colors.Normalize(vmin=torch.amin(images[0]).item(), vmax=torch.amax(images[0]).item())
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            fig.colorbar(sm, ax=ax, label='Z Height or Pixel value')

        fig.tight_layout()

    # plotting multiple images
    else:  # len(images) > 1:
        minmin = min([im.min().item() for im in images])
        maxmax = max([im.max().item() for im in images])

        for i in range(len(images)):
            if isinstance(images[i], np.ndarray):
                img_tensor = tvF.to_tensor(images[i])
            else:
                img_tensor = images[i].detach()

            # img_pil = tvF.to_pil_image(rescale_tensor(img_tensor, as_image=True) if img_tensor.is_floating_point() else img_tensor.to(torch.uint8))

            if titles is not None:
                ax[i].set_title(titles[i], pad=11)
            ax[i].axis('off')

            if colorscale == 'compare':
                ax[i].imshow(img_tensor.squeeze().cpu().numpy(), cmap=cmap, vmin=minmin, vmax=maxmax)
            else:
                ax[i].imshow(img_tensor.squeeze().cpu().numpy(), cmap=cmap)

            if colorscale is True:  # scale for each
                cax_i = make_axes_locatable(ax[i]).append_axes("right", size="5%", pad=0.1)
                norm_i = mpl.colors.Normalize(vmin=img_tensor.min().item(), vmax=img_tensor.max().item())
                sm_i = plt.cm.ScalarMappable(cmap=cmap, norm=norm_i)

                cbar = fig.colorbar(sm_i, cax=cax_i)
                if i == len(images) - 1:
                    cbar.set_label('Z Height or Pixel value', labelpad=12)

        if colorscale == 'compare':
            norm_i = mpl.colors.Normalize(vmin=minmin, vmax=maxmax)
            sm_i = plt.cm.ScalarMappable(cmap=cmap, norm=norm_i)

            pos = ax[-1].get_position()
            left = pos.x0 + pos.width * (1 + 0.05)
            bottom = pos.y0
            width = 0.07 * pos.width
            height = pos.height

            cbar = fig.colorbar(sm_i, cax=fig.add_axes((left, bottom, width, height)))
            cbar.set_label('Z Height or Pixel value', labelpad=12)
        
        # adjust spacing based on requested colorbar, if any
        if colorscale != True:
            plt.subplots_adjust(left=0.01,
                                bottom=0.01,
                                top=0.96,
                                right=0.85 if colorscale else 0.94,
                                wspace=0.01 * len(images),
                                hspace=0.0)
        else:
            fig.tight_layout()

    # saving image montage
    if save_dir != '':
        save_name = f"{os.path.splitext(f_name)[0] if any(ext in f_name for ext in SUPPORTED) else f_name}{dt.today().strftime('%H%M') + '_montage' if not f_name else ''}{'-' + tag if tag else ''}.png"
        fig.savefig(os.path.join(save_dir, save_name), bbox_inches='tight')

    if show:
        plt.ion()
        plt.show()
    else:
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
    ext = os.path.splitext(filepath)[-1].lower()

    if ext not in SUPPORTED:
        raise TypeError(f'File must be in the following supported formats: {SUPPORTED}')

    if ext.lower() in PICTURE_TYPES:
        with Image.open(filepath) as im_pil:
            if im_pil.mode == 'RGBA':       # torchvision Grayscale() can't handle more than 3 channels
                im_pil = im_pil.convert("RGB")
            try:        # apparently in older versions of torchvision, this will fail with 16-bit grayscale images bc it can't convert from dtype uint16
                im_tensor = tvF.pil_to_tensor(im_pil)
            except TypeError:
                im_tensor = tvF.pil_to_tensor(im_pil.convert("I")).to(torch.float64)
        im_tensor = trf.Grayscale()(rescale_tensor(im_tensor.to(torch.float64)))

    else:  # ext.lower() in xyz:
        
        if ext.lower() in ['.pt']:
            im_tensor = torch.load(filepath)
            
        else:
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
