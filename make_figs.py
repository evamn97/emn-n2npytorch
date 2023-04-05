import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def create_fig(montage_fpath, psnr_fpath, title, w_size, p_params):
    with Image.open(montage_fpath).convert('RGB') as montage:
        montage.load()

    trend_df = pd.read_csv(psnr_fpath, sep=',').sort_values(by=['input'])
    trend_df['improvement (ratio)'] = np.divide(np.subtract(trend_df['result'].values, trend_df['input'].values), trend_df['result'])
    trend_df['improvement (binned)'] = trend_df['improvement (ratio)'].rolling(w_size, min_periods=1).mean()

    fig1 = plt.figure(dpi=200, figsize=(13, 10))
    ax1 = fig1.add_subplot(211)
    ax1.imshow(montage)
    ax1.set_xticks([])
    ax1.set_yticks([])
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.set_xlabel('      Input                      Output                Ground Truth')
    ax1.xaxis.set_label_position('top')
    ax1.set_ylabel(title)

    ax2 = fig1.add_subplot(212)
    ax2.scatter(trend_df['input'].values, trend_df['improvement (ratio)'].values,
                s=p_params[0], alpha=0.8, label='Improvement (ratio) vs. Input (dB)')
    ax2.plot(trend_df['input'].values, trend_df['improvement (binned)'].values,
             c='black', linewidth=p_params[1], label='Rolling Mean Fit (window={})'.format(w_size))
    ax2.set(xlabel='Input (dB)')
    ax2.legend(loc='lower left')

    plt.subplots_adjust(left=0.05,
                        bottom=0.08,
                        right=0.99,
                        top=0.95,
                        wspace=0.02,
                        hspace=0.1)
    plt.show()


def combo_fig(montage_list, psnr_list, title_list, w_size, p_params):
    assert len(montage_list) == len(psnr_list) and len(montage_list) == len(title_list), 'Length mismatch: Each montage must have a corresponding psnr plot and title'

    fig, axes = plt.subplots(len(montage_list), 2, dpi=200, figsize=(29, 5 * len(montage_list)))

    for i in range(len(montage_list)):
        with Image.open(montage_list[i]).convert('RGB') as montage:
            montage.load()
        expanded_title = title_list[i] + ' Input dB vs Improvement ratio'

        axes[i, 0].imshow(montage)
        axes[i, 0].set_ylabel(title_list[i], labelpad=0)

        # axes[i, 0].axis('off')
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        for spine in axes[i, 0].spines.values():
            spine.set_visible(False)

        trend_df = pd.read_csv(psnr_list[i], sep=',').sort_values(by=['input'])
        trend_df['improvement (ratio)'] = np.divide(np.subtract(trend_df['result'].values, trend_df['input'].values), trend_df['result'])
        trend_df['improvement (binned)'] = trend_df['improvement (ratio)'].rolling(w_size, min_periods=1).mean()
        axes[i, 1].scatter(trend_df['input'].values, trend_df['improvement (ratio)'].values,
                           s=p_params[0], alpha=0.8, label='Improvement (ratio) vs. Input (dB)')
        axes[i, 1].plot(trend_df['input'].values, trend_df['improvement (binned)'].values,
                        c='black', linewidth=p_params[1], label='Rolling Mean Fit (window={})'.format(w_size))
        axes[i, 1].legend(loc='upper right')

    axes[-1, 0].set_xlabel('    Input                        Output                  Clean Target')
    axes[-1, 1].set_xlabel('Input (dB)')

    # fig.tight_layout()
    # set the spacing between subplots
    plt.subplots_adjust(left=0.0,
                        bottom=0.08,
                        right=0.99,
                        top=0.99,
                        wspace=0.01,
                        hspace=0.1)
    # plt.savefig('pub_figures/noisy-results.png')
    plt.show()


def uneven_combo_fig(montage_list, psnr_list, title_list, w_size, p_params, stacked=True):
    assert len(montage_list) == len(psnr_list) and len(montage_list) == len(title_list), 'Length mismatch: Each montage must have a corresponding psnr plot and title'
    fig = plt.figure(dpi=200, figsize=(29, 5 * len(montage_list)))
    if stacked:
        axes = np.empty((len(montage_list) + len(psnr_list)), dtype=object)
    else:
        axes = np.empty((len(montage_list) + 1), dtype=object)

    cols = 2

    # place montage axes (odd indices)
    left_rows = len(montage_list)
    for i in range(left_rows):
        p = 2 * i + 1
        axes[i] = fig.add_subplot(left_rows, cols, p)
        with Image.open(montage_list[i]).convert('RGB') as montage:
            montage.load()

        axes[i].imshow(montage)
        axes[i].set_ylabel(title_list[i], labelpad=0)

        axes[i].set_xticks([])
        axes[i].set_yticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)

    # place trend axes
    right_rows = len(psnr_list)
    if stacked:
        for i in range(right_rows):
            j = i + left_rows
            q = 2 * i + 2
            axes[j] = fig.add_subplot(right_rows, cols, q)
            trend_df = pd.read_csv(psnr_list[i], sep=',').sort_values(by=['input'])
            trend_df['improvement (ratio)'] = np.divide(np.subtract(trend_df['result'].values, trend_df['input'].values), trend_df['result'])
            trend_df['improvement (binned)'] = trend_df['improvement (ratio)'].rolling(w_size, min_periods=1).mean()
            axes[j].scatter(trend_df['input'].values, trend_df['improvement (ratio)'].values,
                            s=p_params[0], alpha=0.8, label='Improvement (ratio) vs. Input (dB)')
            axes[j].plot(trend_df['input'].values, trend_df['improvement (binned)'].values,
                         c='black', linewidth=p_params[1], label='Rolling Mean Fit (window={})'.format(w_size))
            axes[j].legend(loc='lower left')
    else:
        q = 2
        axes[-1] = fig.add_subplot(1, cols, q)
        for i in range(right_rows):
            trend_df = pd.read_csv(psnr_list[i], sep=',').sort_values(by=['input'])
            trend_df['improvement (ratio)'] = np.divide(np.subtract(trend_df['result'].values, trend_df['input'].values), trend_df['result'])
            trend_df['improvement (binned)'] = trend_df['improvement (ratio)'].rolling(w_size, min_periods=1).mean()
            axes[-1].scatter(trend_df['input'].values, trend_df['improvement (ratio)'].values,
                             s=p_params[0], alpha=0.8, label=f'Improvement (ratio) vs. Input (dB) - {title_list[i]}')
            axes[-1].plot(trend_df['input'].values, trend_df['improvement (binned)'].values,
                          c='black', linewidth=p_params[1], label=f'Rolling Mean Fit (window={w_size})')
            axes[-1].legend(loc='lower left')

    axes[left_rows - 1].set_xlabel('    Input                          Output                    Ground Truth')  # set montage labels
    axes[-1].set_xlabel('Input (dB)')  # set trend labels
    plt.subplots_adjust(left=0.02,
                        bottom=0.09,
                        right=0.99,
                        top=0.99,
                        wspace=0.05,
                        hspace=0.01)
    plt.show()


if __name__ == '__main__':
    plt.rcParams["font.family"] = "serif"

    # # for popup matplotlib windows:
    # plt.rc('font', size=6)
    # plt.rc('axes', titlesize=12)
    # plt.rc('legend', fontsize=10)
    # plot_params = [3, 2]   # [marker_size, line_width]

    # for SciView plots:
    plt.rc('font', size=18)
    plt.rc('axes', labelsize=30)
    plt.rc('legend', fontsize=24)
    plot_params = [24, 4]  # [marker_size, line_width]

    window_size = 15
    titles = ['Bernoulli Noise',
              'Gradient Noise',
              'Low Resolution',
              'Fast Scan sample 1',
              'Fast Scan sample 2']

    psnr_files = ["results/hs20mg-dbtest-bernoulli/psnr.txt",
                  "results/hs20mg-dbtest-gradient/psnr.txt",
                  "results/hs20mg-dbtest-lower/psnr.txt",
                  "psnr-hs20mg-raw.txt",
                  "psnr-tgx-raw.txt"]
    montage_files = ["pub_figures/HS20MG_holes0000-bernoulli0.4-montage.png",
                     "pub_figures/HS20MG_holes0000-gradient0.4-montage.png",
                     "pub_figures/HS20MG_holes0000-lower0.4-montage.png",
                     "pub_figures/HS20MG_holes0000-raw-montage.png",
                     "pub_figures/TGX11Calibgrid_210701_152615-raw0.4-montage.png"]
    cropped_montage_files = ["pub_figures/HS20MG_holes0000-bernoulli0.4-montage-cropped.png",
                             "pub_figures/HS20MG_holes0000-gradient0.4-montage-cropped.png",
                             "pub_figures/HS20MG_holes0000-lower0.4-montage-cropped.png",
                             "pub_figures/HS20MG_holes0017-raw0.12-montage-cropped.png",
                             "pub_figures/TGX110003-raw0.15-montage-cropped.png"]

    # create_fig(cropped_montage_files[3], psnr_files[3], titles[3], window_size, plot_params)

    # combo_fig(cropped_montage_files[:3], psnr_files[:3], titles[:3], window_size, plot_params)

    # uneven_combo_fig(cropped_montage_files[-2:], psnr_files[-2:], titles[-2:], window_size, plot_params, stacked=True)
    combo_fig(cropped_montage_files[-2:], psnr_files[-2:], titles[-2:], window_size, plot_params)

    # trend_df = pd.read_csv(psnr_files[1], sep=',').sort_values(by=['input'])
    # trend_df['improvement (ratio)'] = np.divide(np.subtract(trend_df['result'].values, trend_df['input'].values), trend_df['result'])
