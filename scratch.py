from time import sleep

import matplotlib.pyplot as plt
import pandas as pd

from src.data_prep import *

if __name__ == '__main__':
    sleep(1)

    # ---------------------------------------------------------------- DATA AUGMENTING ----------------------------------------------------------------
    # data_root = "C:/Users/eva_n/OneDrive - The University of Texas at Austin/SANDIA PHD RESEARCH/Ryan-AFM-Data/"
    # n2npt_root = "C:/Users/eva_n/OneDrive - The University of Texas at Austin/PyCharm Projects/emn-n2n-pytorch/"
    #
    # source_in_dir = os.path.join(data_root, "TGX11_12um20um_02-02-23/xyz-conversions")
    # target_in_dir = os.path.join(data_root, "TGX11_12um20um_02-02-23/xyz-processed")
    # source_out_dir = os.path.join(n2npt_root, "tgx2_xyz_data")
    #
    # number = 1000
    # px = 256
    # m_angle = 300
    #
    # # augment(source_in_dir, source_out_dir, number, min_px=px, max_angle=m_angle)
    # augment_pairs(source_in_dir, source_out_dir, target_in_dir, number, min_px=px, max_angle=m_angle)
    #
    # split_ratio = 0.8
    # split(source_out_dir, split_ratio)
    #
    # nt = 44
    # test_out_dir = os.path.join(source_out_dir, "test")
    # get_test(source_in_dir, test_out_dir, nt, target_in_dir)
    # .................................................................................................................................................

    # ------------------------------------------------------------- VALID STATS PLOTTING --------------------------------------------------------------
    ckpt_dir = "ckpts/xyz-raw/xyz-raw0.4l2/"

    f_name = os.path.join(ckpt_dir, "n2n-stats.json")
    stats = pd.read_json(f_name)
    stats['psnr_over_loss'] = stats['valid_psnr'] / stats['valid_loss']

    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots()
    ax.plot(stats['psnr_over_loss'])
    ax.set(title='Valid PSNR / Valid Loss', xlabel='Epoch')
    fig.tight_layout()
    plt.show()
    save_name = os.path.join(ckpt_dir, 'psnr-over-loss.png')
    plt.savefig(save_name, dpi=200)
    # .................................................................................................................................................

    # ---------------------------------------------------------------- RENAMING FILES -----------------------------------------------------------------
    # path = os.path.join(n2npt_root, "bash-script-files")
    # mode = 'ext'
    # replacing = ''
    # new_string = '.sh'
    # batch_rename(path, mode, new_string, to_replace=replacing)
    # .................................................................................................................................................

    # ---------------------------------------------------------- DROPPING X&Y FROM XYZ DATA -----------------------------------------------------------
    # xyz_dir = os.path.join(data_root, "xyz-extra-processed")
    # out_dir = os.path.join(data_root, "z-only-extra-processed")
    # if not os.path.isdir(out_dir):
    #     os.mkdir(out_dir)
    # with os.scandir(xyz_dir) as folder:
    #     for file in folder:
    #         name = os.path.splitext(file.name)[0]
    #         save_path = os.path.join(out_dir, (name + '.csv'))
    #         z = xyz_to_zfield(file.path)
    #         arr3d_to_xyz(z, save_path)
    # print("Done!")
    # .................................................................................................................................................

    # ----------------------------------------------------- ImgRec input/output trends plotting -----------------------------------------------------
    # window_size = 15
    #
    # bernoulli_db = pd.read_csv("results/hs20mg-dbtest-bernoulli/psnr.txt", sep='\t').sort_values(by=['input'])
    # bernoulli_db['improvement (ratio)'] = np.divide(np.subtract(bernoulli_db['result'].values, bernoulli_db['input'].values), bernoulli_db['result'])
    # bernoulli_db['improvement (binned)'] = bernoulli_db['improvement (ratio)'].rolling(window_size, min_periods=1).mean()
    #
    # lower_db = pd.read_csv("results/hs20mg-dbtest-lower/psnr.txt", sep='\t').sort_values(by=['input'])
    # lower_db['improvement (ratio)'] = np.divide(np.subtract(lower_db['result'].values, lower_db['input'].values), lower_db['result'])
    # lower_db['improvement (binned)'] = lower_db['improvement (ratio)'].rolling(window_size, min_periods=1).mean()
    #
    # gradient_db = pd.read_csv("results/hs20mg-dbtest-gradient/psnr.txt", sep='\t').sort_values(by=['input'])
    # gradient_db['improvement (ratio)'] = np.divide(np.subtract(gradient_db['result'].values, gradient_db['input'].values), gradient_db['result'])
    # gradient_db['improvement (binned)'] = gradient_db['improvement (ratio)'].rolling(window_size, min_periods=1).mean()
    #
    # raw_db = pd.read_csv("psnr-hs20mg-raw.txt", sep=',').sort_values(by=['input'])
    # raw_db['improvement (ratio)'] = np.divide(np.subtract(raw_db['result'].values, raw_db['input'].values), raw_db['result'])
    # raw_db['improvement (binned)'] = raw_db['improvement (ratio)'].rolling(window_size, min_periods=1).mean()
    #
    # titles = ['Bernoulli Noise Input dB vs Improvement ratio',
    #           'Low Resolution Input dB vs Improvement ratio',
    #           'Gradient Noise Input dB vs Improvement ratio',
    #           'Raw to Processed Input dB vs Improvement ratio']
    #
    # fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    # dfs = [bernoulli_db, lower_db, gradient_db, raw_db]
    # assert len(axes.flat) == len(dfs), "The number of subplots must be equal to the number of dataframes"
    # for i in range(axes.shape[0]):
    #     for j in range(axes.shape[1]):
    #         axes[i, j].scatter(dfs[i * 2 + j]['input'].values, dfs[i * 2 + j]['improvement (ratio)'].values, s=5, alpha=0.65, label='Trend Points')
    #         axes[i, j].plot(dfs[i * 2 + j]['input'].values, dfs[i * 2 + j]['improvement (binned)'].values, c='blue', linewidth=2, label='Rolling Mean Fit \n(window={})'.format(window_size))
    #         axes[i, j].set(title=titles[i * 2 + j], xlabel='Input dB', ylabel='Improvement ratio')
    #         axes[i, j].legend(loc='upper right')
    # fig.tight_layout()
    # plt.show()
    # fig.savefig("pub_figures/db_trends.png")
    # plt.close()
    #
    # plt.rc('font', size=30)
    # plt.rc('axes', titlesize=44)
    #
    # fig1, ax1 = plt.subplots(figsize=(29, 10))
    # ax1.scatter(bernoulli_db['input'].values, bernoulli_db['improvement (ratio)'].values, s=20, alpha=0.8, label='Trend Points')
    # ax1.plot(bernoulli_db['input'].values, bernoulli_db['improvement (binned)'].values, c='blue', linewidth=4, label='Rolling Mean Fit \n(window={})'.format(window_size))
    # ax1.set(title=titles[0], xlabel='Input dB', ylabel='Improvement ratio')
    # ax1.legend(loc='upper right')
    # fig1.tight_layout()
    # plt.show()
    # plt.close()
    #
    # fig2, ax2 = plt.subplots(figsize=(29, 10))
    # ax2.scatter(lower_db['input'].values, lower_db['improvement (ratio)'].values, s=20, alpha=0.8, label='Trend Points')
    # ax2.plot(lower_db['input'].values, lower_db['improvement (binned)'].values, c='blue', linewidth=4, label='Rolling Mean Fit \n(window={})'.format(window_size))
    # ax2.set(title=titles[1], xlabel='Input dB', ylabel='Improvement ratio')
    # ax2.legend(loc='upper right')
    # fig2.tight_layout()
    # plt.show()
    # plt.close()
    #
    # fig3, ax3 = plt.subplots(figsize=(29, 10))
    # ax3.scatter(gradient_db['input'].values, gradient_db['improvement (ratio)'].values, s=20, alpha=0.8, label='Trend Points')
    # ax3.plot(gradient_db['input'].values, gradient_db['improvement (binned)'].values, c='blue', linewidth=4, label='Rolling Mean Fit \n(window={})'.format(window_size))
    # ax3.set(title=titles[2], xlabel='Input dB', ylabel='Improvement ratio')
    # ax3.legend(loc='upper right')
    # fig3.tight_layout()
    # plt.show()
    # plt.close()
    #
    # fig4, ax4 = plt.subplots(figsize=(29, 10))
    # ax4.scatter(raw_db['input'].values, raw_db['improvement (ratio)'].values, s=20, alpha=0.8, label='Trend Points')
    # ax4.plot(raw_db['input'].values, raw_db['improvement (binned)'].values, c='blue', linewidth=4, label='Rolling Mean Fit \n(window={})'.format(window_size))
    # ax4.set(title=titles[3], xlabel='Input dB', ylabel='Improvement ratio')
    # ax4.legend(loc='upper right')
    # fig4.tight_layout()
    # plt.show()
    # plt.close()

    # .................................................................................................................................................
