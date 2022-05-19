import numpy as np
import pandas as pd
from numpy import random as rand
import torchvision.transforms as trf
from PIL import Image, ImageDraw
import random
from math import pi, cos, sin, ceil


def create_image(img, p, crop, style='r'):
    if crop > 0:
        tcrop = trf.RandomCrop((crop, crop))  # create random cropper
        cropped_img_PIL = tcrop.forward(img)  # use cropper to crop image
        ground_truth_array = np.array(cropped_img_PIL)
    else:
        ground_truth_array = np.array(img)
        cropped_img_PIL = img
    if len(ground_truth_array.shape) == 3:
        h, w, c = ground_truth_array.shape  # get cropped array dimensions
    else:
        h = ground_truth_array.shape[0]
        w = ground_truth_array.shape[1]
        c = 1

    rng = rand.default_rng()

    if style == 'l':      # lowers resolution of input image (should be used with clean targets)
        temp_img = cropped_img_PIL
        hN, wN = int(h*p), int(w*p)     # choose new dims based on p value
        lowered = trf.Resize((hN, wN)).forward(temp_img)
        resized = trf.Resize((h, w)).forward(lowered)
        img_array_noised = np.array(resized)

    elif style == 'nu':     # random noise up to +/- 10 percent values (not binary)
        ran = (np.max(ground_truth_array) - np.min(ground_truth_array))*0.1
        rand_noise = rand.uniform(-1*ran, ran, (h, w))
        bin_mask = rng.binomial(1, p, (h, w))
        mask = np.multiply(rand_noise, bin_mask)
        if c > 1:
            m_list = []
            for dim in range(c):
                m_list.append(mask)
            mask = np.dstack(m_list)
        img_array_noised = np.add(ground_truth_array, mask)

    elif style == 'g':     # create a noise gradient from left to right
        ran = (np.max(ground_truth_array) - np.min(ground_truth_array))*p     # use 0.1 here for +/- 10% noise; this is arbitrary
        rand_noise = rand.uniform(-1, 1, (h, w))
        grad = np.tile(np.linspace(0, ran, num=w), (h, 1))
        mask = np.multiply(rand_noise, grad)
        if c > 1:
            m_list = []
            for dim in range(c):
                m_list.append(mask)
            mask = np.dstack(m_list)
        img_array_noised = np.add(ground_truth_array, mask)

    elif style == 'msd':  # moving standard deviation (I chose N = 3) to mimic variable dwell
        if c > 1:
            temp = ground_truth_array[:, :, :3]  # for converting to grayscale, we ignore transparency channel
            ct = 3  # number channels in temp
        else:
            temp = ground_truth_array
            ct = 1

        m_list = []  # for noise to add
        for channel in range(ct):
            layer = temp[:, :, channel]

            i = 0
            for row in layer:
                if i % 2 != 0:  # flips odd numbered rows to mimic zig-zag scanning pattern (will flip back later)
                    row = np.flip(row)
                    layer[i] = row
                i += 1
            s = pd.Series(layer.flatten())  # convert to pandas time series to get rolling stdev
            stdev_s = np.nan_to_num(np.asarray(s.rolling(5).std()))  # take rolling stdev, NaN->0, convert back to numpy
            stdev_s = np.reshape(stdev_s, (h, w))  # reshape flattened array

            i = 0
            for row in stdev_s:
                if i % 2 != 0:  # flips odd numbered rows back around to preserve image
                    row = np.flip(row)
                    stdev_s[i] = row
                i += 1
            scale = np.amax(ground_truth_array)
            noise_s = np.multiply(np.nan_to_num(1 / (stdev_s + 1)), p * scale)

            signs = rng.binomial(1, 0.5, (h, w))
            signs[signs == 0] = -1
            noise_s = np.multiply(noise_s, signs)     # since stdev is always pos, but we want a +/- chance

            m_list.append(noise_s)

        if ct > 1:      # check if temp is grayscale (ct = 1) or RGB (ct = 3)
            if c > 3:
                m_list.append(ground_truth_array[:, :, -1])     # if original image is RGBA, add back the original transparency layer
            noise_to_add = np.dstack(m_list)
            debug_arr = np.mean(noise_to_add[:, :, :ct], axis=2)    # convert to grayscale for debug test
        else:
            noise_to_add = noise_s
            debug_arr = noise_to_add
        test = Image.fromarray(1 / (debug_arr + 1), mode='L')
        test.show()
        img_array_noised = np.add(noise_to_add, ground_truth_array)

    elif style == 'ma':
        if c > 1:
            temp = ground_truth_array[:, :, :3]  # we ignore transparency channel
            ct = 3  # number channels in temp
        else:
            temp = ground_truth_array
            ct = 1

        m_list = []  # for layers to stack
        for channel in range(ct):
            layer = temp[:, :, channel]

            i = 0
            for row in layer:
                if i % 2 != 0:  # flips odd numbered rows to mimic zig-zag scanning pattern (will flip back later)
                    row = np.flip(row)
                    layer[i] = row
                i += 1
            s = pd.Series(layer.flatten())  # convert to pandas time series to get rolling avg
            avg_s = np.nan_to_num(np.asarray(s.rolling(4, min_periods=1).mean()))  # take rolling avg, NaN->0 (should be redundant w/min_periods=1 ?), convert back to numpy
            avg_s = np.reshape(avg_s, (h, w))  # reshape flattened array

            i = 0
            for row in avg_s:
                if i % 2 != 0:  # flips odd numbered rows back around to preserve image
                    row = np.flip(row)
                    avg_s[i] = row
                i += 1

            m_list.append(avg_s)

        if ct > 1:
            if c > 3:
                m_list.append(ground_truth_array[:, :, -1])
            mavg_img = np.dstack(m_list)
            debug_arr = np.mean(mavg_img[:, :, :ct], axis=2)    # grayscale debug test
        else:
            mavg_img = avg_s    # TODO: where could it be undefined??

        img_array_noised = mavg_img     # TODO: is this right?

    else:   # style == 'r':  # random trials
        r_mask = rng.binomial(1, p, (h, w))
        if c > 1:
            m_list = []
            for dim in range(c):
                m_list.append(r_mask)
            mask = np.dstack(m_list)
        else:
            mask = r_mask
        img_array_noised = np.multiply(ground_truth_array, mask)

    return np.clip(img_array_noised, 0, 255).astype(np.uint8)


p = 0.01
crop_size = 512
example = Image.open("new_test_data\\512\\HS-20MG_00051.png")
# example = Image.open("new_test_data\\128\\HS-20MG_00003.png")
# example = Image.open("new_test_data\\TGX11Calibgrid_210701_152615.png")
corrupt_array = create_image(example, p=p, style='ma', crop=crop_size)
corrupted = trf.ToPILImage()(corrupt_array)
# corrupted.save('resized_test.png')
example.show()
corrupted.show()
