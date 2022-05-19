from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw
# from math import cos, sin, pi, ceil
import torchvision.transforms as trf
import torchvision.transforms.functional as tvF
import os
import random
from datasets import load_dataset
from noise2noise import Noise2Noise
from argparse import ArgumentParser
from utils import create_montage


# #
# # date = f'{datetime.now():%H%M%S}'
# p = 0.5
# crop = 128
# example = Image.open("..\\images\\png-conversions\\TGX11Calibgrid_210701_151709.png")
# corrupt_array = create_image(example, p=p, crop=128, style='b')
# corrupt_array = np.clip(corrupt_array, 0, 255).astype(np.uint8)
# corrupted = Image.fromarray(corrupt_array)
# corrupted.show()
# block_line = np.concatenate([np.zeros(round(p * 10)), np.ones(round((1 - p) * 10))]).astype(int)
# if crop > 10:
#     while len(block_line) < crop:
#         block_line = np.append(block_line, block_line)
#     block_line = block_line[:crop]
# x, y = np.meshgrid(block_line, block_line)
# blocks = np.logical_not(np.bitwise_or(x, y)).astype(int)
#
# -------------------------------------------------------------------------------------------------------------------- #
# crop = 129
# grid1 = np.array([[0, 1, 0, 1], [1, 1, 1, 1], [0, 1, 0, 1], [1, 1, 1, 1]])
# grid2 = np.array([[1, 1, 1, 1], [1, 0, 1, 0], [1, 1, 1, 1], [1, 0, 1, 0]])
# choice = random.choice([grid1, grid2])
# if crop % 4 == 0:
#     grid = np.tile(choice, (int(crop/4), int(crop/4)))
# else:
#     grid = np.tile(choice, (int(crop/4) + 1, int(crop/4) + 1))
#     grid = grid[:crop, :crop]

# -------------------------------------------------------------------------------------------------------------------- #
# def polar_to_cartesian(r, theta):
#     return int(r*cos(theta)), int(r*sin(theta))
#
#
# def translate(point, screen_size):
#     """
#     Takes a point and converts it to the appropriate coordinate system.
#     Note that PIL uses upper left as 0, we want the center.
#     Args:
#         point (real, real): A point in space.
#         screen_size (int): Size of an N x N screen.
#     Returns:
#         (real, real): Translated point for Pillow coordinate system.
#     """
#     return point[0] + screen_size / 2, point[1] + screen_size / 2
#
#
# def draw_spiral(b, img, step=0.1, loops=10, w=0, a=1):
#     """
#     Draw the Archimedean spiral defined by:
#     r = a + b*theta
#     Args:
#         a (real): First parameter (permanently set to 1 here)
#         b (real): Second parameter
#         img (Image): Image to write spiral to.
#         step (real): How much theta should increment by. (default: 0.5)
#         loops (int): How many times theta should loop around. (default: 5)
#         w (int): width of line (default: 0)
#     """
#     draw = ImageDraw.Draw(img)
#     theta = 0.0
#     r = a
#     prev_pos = polar_to_cartesian(r, theta)
#     while theta < 2 * loops * pi:
#         theta += step / r
#         r = a + b * theta
#         # Draw pixels, but remember to convert to Cartesian:
#         pos = polar_to_cartesian(r, theta)
#         draw.line(translate(prev_pos, img.size[0]) +
#                   translate(pos, img.size[0]), fill=1, width=w)
#         prev_pos = pos
#
#
# dim = 512
# IMAGE_SIZE = dim, dim
# # b = random.uniform(0.0, 8.0)
# # loop = ceil(14.4934476*(1 / (b**1.0295671)))    # best guess power fit for choosing num of loops
# # line_dict = {'1': (False, 1), '2': (False, 2), '3': (False, 3), '4': (False, 4), '5': (False, 5), '5i': (True, 5),
# #              '4i': (True, 4), '3i': (True, 3), '2i': (True, 2), '1i': (True, 1)}
# # invert, line = random.choice(list(line_dict.values()))
# # image = Image.new('1', IMAGE_SIZE)
# # draw_spiral(b, image, w=line, loops=loop)
# # smask = np.array(image).astype(int)
# # if invert:
# #     smask = 1 - smask
#
# for b in np.arange(0.3, 3, 0.1):
#     loop = 140
#     while loop > 58:
#         image = Image.new('1', IMAGE_SIZE)
#         draw_spiral(b, image, loops=loop, step=5)
#         print("b = " + repr(b) + ", loops = " + repr(loop))
#         image.show()
#         loop = loop - 4
# -------------------------------------------------------------------------------------------------------------------- #
#
# elif style == 's':  # spiral mask
#
#
# def polar_to_cartesian(r, theta):
#     return int(r * cos(theta)), int(r * sin(theta))
#
#
# def translate(point, screen_size):
#     """
#     Takes a point and converts it to the appropriate coordinate system.
#     Note that PIL uses upper left as 0, we want the center.
#     Args:
#         point (real, real): A point in space.
#         screen_size (int): Size of an N x N screen.
#     Returns:
#         (real, real): Translated point for Pillow coordinate system.
#     """
#     return point[0] + screen_size / 2, point[1] + screen_size / 2
#
#
# def draw_spiral(b, spiral_img, step=0.1, loops=10, line_width=0):
#     """
#     Draw the Archimedean spiral defined by:
#     r = a + b*theta
#     Args:
#         a (real): First parameter (permanently set to 1 here)
#         b (real): Second parameter
#         spiral_img (Image): Image to write spiral to.
#         step (real): How much theta should increment by. (default: 0.5)
#         loops (int): How many times theta should loop around. (default: 5)
#         line_width (int): width of line (default: 0)
#     """
#     draw = ImageDraw.Draw(spiral_img)
#     theta = 0.0
#     r = 1
#     prev_pos = polar_to_cartesian(r, theta)
#     while theta < 2 * loops * pi:
#         theta += step / r
#         r = 1 + b * theta
#         # Draw pixels, but remember to convert to Cartesian:
#         pos = polar_to_cartesian(r, theta)
#         draw.line(translate(prev_pos, spiral_img.size[0]) +
#                   translate(pos, spiral_img.size[0]), fill=1, width=line_width)
#         prev_pos = pos
#
#
# IMAGE_SIZE = w, h
# b = rand.uniform(0.0, 8.0)
# loop = ceil(14.4934476 * (1 / (b ** 1.0295671)))  # best guess power fit for choosing num of loops
# line_dict = {'1': (False, 1), '2': (False, 2), '3': (False, 3), '4': (False, 4), '5': (False, 5),
#              '5i': (True, 5),
#              '4i': (True, 4), '3i': (True, 3), '2i': (True, 2), '1i': (True, 1)}
# invert, line = rand.choice(list(line_dict.values()))
# image = Image.new('1', IMAGE_SIZE)
# draw_spiral(b, image, line_width=line, loops=loop)
# mask = np.array(image).astype(int)
# if invert:
#     mask = 1 - mask
# img_array_noised = np.multiply(ground_truth_array, mask)

# -------------------------------------------------------------------------------------------------------------------- #

# if style == 'g':  # grid mask
#     preserved_px = round(p * 10)
#     noisy_px = round((1 - p) * 10)
#     block_line = np.concatenate([np.zeros(preserved_px), np.ones(noisy_px)]).astype(int)
#     if w > 10:
#         while len(block_line) < w:
#             switch = rand.choice(np.arange(0, noisy_px + 1))
#             block_add = np.concatenate([np.ones(switch), np.zeros(preserved_px),
#                                         np.ones(noisy_px - switch)]).astype(int)
#             block_line = np.append(block_line, block_add)
#         block_line = block_line[:w]
#     x, y = np.meshgrid(block_line, block_line)
#     bgrid = np.logical_not(np.bitwise_or(x, y)).astype(int)
#     if c > 1:
#         m_list = []
#         for dim in range(c):  # stack array for RGB (3-channel) images
#             m_list.append(bgrid)
#         mask = np.dstack(m_list)
#     else:
#         mask = bgrid
#     img_array_noised = np.multiply(ground_truth_array, mask)

# -------------------------------------------------------------------------------------------------------------------- #


# path = "..\\new_afm_data\\test\\HS-20MG.png"
# p = 0.6
# with Image.open(path) as original:
#     og_tensor = tvF.to_tensor(original)
#     c, h, w = og_tensor.shape
#     hN, wN = int(h * p), int(w * p)  # choose new dims based on p value
#     lowered = trf.Resize((hN, wN)).forward(original)
#     original.show()
# resized = trf.Resize((h, w)).forward(lowered)
# resized.show()

# -------------------------------------------------------------------------------------------------------------------- #


        if c > 1:
            temp = ground_truth_array[:, :, :3]     # for converting to grayscale, we ignore transparency channel
            ct = 3      # number channels in temp
        else:
            temp = ground_truth_array
            ct = 1

        m_list = []     # for noise to add
        s_list = []     # for debugging; creates image out of normalized stdev
        for channel in range(ct):
            layer = temp[:, :, channel]

            i = 0
            for row in layer:
                if i % 2 != 0:  # flips odd numbered rows to mimic zig-zag scanning pattern (will flip back later)
                    row = np.flip(row)
                    layer[i] = row
                i += 1
            s = pd.Series(layer.flatten())      # convert to pandas time series to get rolling stdev
            stdev_s = np.nan_to_num(np.asarray(s.rolling(16).std()))  # take rolling stdev, NaN->0, convert back to numpy
            stdev_s = np.reshape(stdev_s, (h, w))   # reshape flattened array

            i = 0
            for row in stdev_s:
                if i % 2 != 0:  # flips odd numbered rows back around to preserve image
                    row = np.flip(row)
                    stdev_s[i] = row
                i += 1
            scale = np.amax(ground_truth_array)
            noise_s = np.multiply(np.nan_to_num(1 / (stdev_s + 1)), p*scale)

            # signs = rng.binomial(1, 0.5, (h, w))
            # signs[signs == 0] = -1
            # noise_s = np.multiply(noise_s, signs)     # since stdev is always pos, but we want a +/- chance

            m_list.append(noise_s)
            s_list.append(stdev_s)


        if ct > 1:
            if c > 3:
                m_list.append(np.zeros((h, w)))
            noise_to_add = np.dstack(m_list)
            stdev_arr = np.mean(np.dstack(s_list), axis=2)
        else:
            noise_to_add = noise_s
            stdev_arr = stdev_s
        test = Image.fromarray(1 / (stdev_arr + 1), mode='L')
        test.show()
        img_array_noised = np.add(noise_to_add, ground_truth_array)