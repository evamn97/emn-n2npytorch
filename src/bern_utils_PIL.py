import numpy as np
from numpy import random as rand
import torchvision.transforms as trf
from math import sin, cos, pi, ceil
from PIL import Image, ImageDraw


def create_image(img, p=0.5, crop=0, style='r'):
    if crop > 0:
        tcrop = trf.RandomCrop((crop, crop))  # create random cropper
        cropped_img_PIL = tcrop.forward(img)  # use cropper to crop image
        ground_truth_array = np.array(cropped_img_PIL)
    else:
        ground_truth_array = np.array(img)
        cropped_img_PIL = img
    if len(ground_truth_array.shape) == 3:
        w, h, c = ground_truth_array.shape  # get cropped array dimensions
    else:
        w = ground_truth_array.shape[0]
        h = ground_truth_array.shape[1]
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

    return img_array_noised
