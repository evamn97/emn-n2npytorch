from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from math import cos, sin, pi, ceil
import torchvision.transforms as trf
import torchvision.transforms.functional as tvF
import os
import random
from data_prep import random_trf
import matplotlib.pyplot as plt


def normalize(arr):
    """ Normalizes an input array to between 0 and 1.

    :param arr: numpy array input
    :return: normalized array
    """
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def xyz_to_torch(filepath):
    xyz_df = pd.read_csv(filepath, names=['x', 'y', 'z'], delimiter='\t', index_col=False)
    profiles = len(set(xyz_df['y'].values))
    z_field_arr = normalize(xyz_df['z'].values.reshape((-1, profiles)))
    z_field = tvF.to_tensor(z_field_arr)
    return z_field


# input_file = "HS20MG_holes0000.xyz"
# z_tensor = xyz_to_torch(input_file)
# z_img = tvF.to_pil_image(z_tensor)
#
# z1_tensor = random_trf(z_tensor)
# z1_img = tvF.to_pil_image(z1_tensor)
# z2_tensor = random_trf(z_tensor)
# z2_img = tvF.to_pil_image(z2_tensor)


bernoulli_db = pd.DataFrame({'input db': [5.07, 4.52, 4.53, 4.42, 5.37, 4.83, 10.25, 10.13, 10.36, 10.49, 10.59, 10.50],
                             'output db': [39.0, 37.96, 38.64, 37.90, 33.18, 31.71, 31.05, 34.31, 30.92, 30.13, 38.30, 35.36]}
                            ).sort_values(by=['input db'])
lower_db = pd.DataFrame({'input db': [25.56, 25.99, 26.40, 28.63, 28.65, 25.87, 26.50, 28.38, 28.80, 28.08, 26.29, 25.87],
                         'output db': [39.52, 39.77, 39.70, 38.95, 39.53, 38.81, 34.30, 34.40, 34.54, 33.49, 33.59, 33.24]}
                        ).sort_values(by=['input db'])
gradient_db = pd.DataFrame({'input db': [18.54, 18.72, 18.37, 18.85, 18.80, 18.77, 13.80, 14.35, 13.64, 14.49, 14.26, 14.08],
                            'output db': [41.38, 40.79, 39.51, 41.51, 41.05, 39.96, 23.07, 21.51, 23.32, 21.18, 21.36, 22.01]}
                           ).sort_values(by=['input db'])
raw_db = pd.DataFrame({'input db': [10.68, 8.17, 4.90, 7.84, 5.40, 6.34, 9.13, 7.14, 6.44, 9.16, 8.43, 11.78],
                       'output db': [21.41, 24.12, 23.64, 25.24, 30.61, 21.36, 26.03, 20.95, 29.47, 22.54, 17.83, 14.05]}
                      ).sort_values(by=['input db'])

fig, axes = plt.subplots(2, 2, figsize=(20, 12))
titles = ['Bernoulli Corruption Input dB vs Output dB',
          'Low Resolution Input dB vs Output dB',
          'Gradient Noise Input dB vs Output dB',
          'Raw AFM to Processed Data Input dB vs Output dB']
dfs = [bernoulli_db, lower_db, gradient_db, raw_db]
assert len(axes.flat) == len(dfs), "The number of subplots must be equal to the number of dataframes"

for i in range(axes.shape[0]):
    for j in range(axes.shape[1]):
        axes[i, j].plot(dfs[i * 2 + j]['input db'].values, dfs[i * 2 + j]['output db'].values, '.-')
        axes[i, j].set(title=titles[i * 2 + j], xlabel='Input dB', ylabel='Output dB')
fig.tight_layout()
plt.show()
fig.savefig("db_trends.png")
