import numpy as np
import torch as t
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from bernoulli_utils import create_image
from PIL import Image
from matplotlib import pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

black = Image.new('RGB', (512, 512), (10, 10, 10))
black_tensor = tvF.to_tensor(black)
white = Image.new('RGB', (512, 512), (255, 255, 255))
white_tensor = tvF.to_tensor(white)


def psnr(corrupt_input, clean_target):
    """ Computes peak signal-to-noise ratio. """
    # mse = F.mse_loss(corrupt_input, clean_target, reduction='sum')
    # inverse_mse = 1 / mse
    # log = t.log10(inverse_mse)
    # peak_snr = 10 * log
    dims = list(clean_target.shape)
    if len(dims) > 2:
        R = 255
    else:
        R = 1
    peak_snr = 10 * t.log10(R ** 2 / F.mse_loss(corrupt_input, clean_target))
    return peak_snr


target = "new_test_data\\256\\HS-20MG.png"
source = "ml_results\\denoised-gradient-2021-12-19_1846\\HS-20MG-gradient-noisy.png"
denoised = "ml_results\\denoised-gradient-2021-12-19_1846\\HS-20MG-gradient-denoised.png"

with Image.open(target) as T:
    target_t = tvF.to_tensor(T)[0:3, :, :]
target_img = tvF.to_pil_image(target_t)
with Image.open(source) as S:
    source_t = tvF.to_tensor(S)
source_img = tvF.to_pil_image(source_t)
with Image.open(denoised) as D:
    denoised_t = tvF.to_tensor(D)
denoised_img = tvF.to_pil_image(denoised_t)

source_l1Loss = F.l1_loss(source_t, target_t)
source_log = t.log10(1 / source_l1Loss)
denoised_l1Loss = F.l1_loss(denoised_t, target_t)
denoised_log = t.log10(1 / denoised_l1Loss)

titles = ['Input: {:.2f} '.format(source_l1Loss.item()), 'Denoised: {:.2f} '.format(denoised_l1Loss.item()),
          'Ground truth']
fig, ax = plt.subplots(1, 3, figsize=(21, 7))
ax[0].imshow(source_img)
ax[0].set_title(titles[0])
ax[1].imshow(denoised_img)
ax[1].set_title(titles[1])
ax[2].imshow(target_img)
ax[2].set_title(titles[2])
ax[0].axis('off')
ax[1].axis('off')
ax[2].axis('off')

plt.show()

with Image.open(target) as T:
    test_noisy = create_image(T, p=0.999, crop=256, style='r')
    test_target = create_image(T, p=1.0, crop=256, style='r')
test_noisy_t = tvF.to_tensor(test_noisy)
test_target_t = tvF.to_tensor(test_target)
loss = F.l1_loss(test_noisy_t, test_target_t)
scaled = (1-loss)*100
test_noisy_img = Image.fromarray(test_noisy, mode='RGBA')
test_target_img = Image.fromarray(test_target, mode='RGBA')

test_noisy_img.show()
# test_target_img.show()
