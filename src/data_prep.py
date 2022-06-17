from typing import Union

import PIL.Image
import numpy as np
import pandas as pd
import random
import torch
import torchvision.transforms as trf
from torchvision.transforms import functional as tvF
import PIL.Image as Image
import os
from tqdm import tqdm
from pathos.helpers import cpu_count
from pathos.pools import ProcessPool as Pool
import shutil
from time import sleep


def normalize(arr, as_image=False):
    """ Normalizes an input array to between 0 and 1.

    :param arr: numpy array input
    :param as_image: changes normalization range to 0-255 and converts to int dtype
    :return: normalized array
    """
    if as_image:
        return (255 * (arr - np.min(arr)) / np.ptp(arr)).astype(int)
    else:
        return (arr - np.min(arr)) / np.ptp(arr)


def highest_pow_2(n):
    p = int(np.log2(n))  # get exponent
    res = int(pow(2, p))  # get result
    return res


def xyz_to_zfield(xyz_filepath, return3d=False):
    """Loads .xyz format data and converts into a 2D numpy array of height (z) values.
        * NOTE: P (# profiles/rows) must be set manually for rotated images!!"""
    df = pd.read_csv(xyz_filepath, names=['x', 'y', 'z'], delimiter='\t', index_col=False)
    P = len(set(df['y'].values))
    try:
        z_arr = df['z'].values.reshape((P, -1))
    except ValueError:
        P = highest_pow_2(P)
        z_arr = df['z'].values.reshape((P, -1))

    if not return3d:
        return z_arr
    else:
        x_arr = df['x'].values.reshape((P, -1))
        y_arr = df['y'].values.reshape((P, -1))
        arr_3d = np.stack((x_arr, y_arr, z_arr), axis=2)
        return arr_3d


def arr3d_to_xyz(arr3d, out_path):
    xyz_df = pd.DataFrame({'x': arr3d[:, :, 0].flat, 'y': arr3d[:, :, 1].flat, 'z': arr3d[:, :, 2].flat})
    xyz_df.to_csv(out_path, header=False, index=False, sep='\t')


def conversions(f_in, new_type):
    supported = [Image.Image, torch.Tensor, np.ndarray]
    if type(f_in) not in supported:
        raise TypeError("Unsupported input type.")
    if new_type not in supported:
        raise TypeError("Unsupported conversion type.")
    assert type(f_in) != new_type, "Input type matches conversion type."

    if new_type == Image.Image:
        if type(f_in) == Image.Image:
            return tvF.to_pil_image(f_in)
        elif type(f_in) == np.ndarray:
            return Image.fromarray(f_in)
    elif new_type == torch.Tensor:
        return tvF.to_tensor(f_in)
    else:  # new_type == np.ndarray
        if type(f_in) == Image.Image:
            return np.array(f_in)
        elif type(f_in) == torch.Tensor:
            return np.rot90(np.rot90(f_in.numpy(), axes=(0, 2)), k=3).squeeze()


def random_trf(image: Union[Image.Image, torch.Tensor, np.ndarray], min_dim=None, target=None, max_angle=270.0):
    """ Applies random transformations to a single image or pair for data augmentation.
    :param image: Independent image, or source image if using pairs.
    :param min_dim: Minimum dimension of input image(s). For transforming sets of differently sized images, locks size of output images.
    :param target: Corresponding target in an image pair.
    :param max_angle: maximum angle range within which to rotate the image.
    """
    if type(image) != torch.Tensor:
        img = tvF.to_tensor(image)
    else:
        img = image
    P = img.shape[1]
    if min_dim is None:
        min_dim = P
    rng = np.random.default_rng()
    if not max_angle <= 0:
        angle = rng.uniform(0, max_angle)  # get random angle in degrees
    else:
        angle = 0
    rad_angle = np.radians(angle)
    c_crop = int(P / (np.abs(np.sin(rad_angle)) + np.abs(np.cos(rad_angle))))  # get bbox size based on rotation
    min_crop = int(min_dim / (2 * np.cos(np.pi / 4)))  # get smallest bbox
    final_crop = highest_pow_2(min_crop)  # must be power of 2
    temp_source = trf.CenterCrop(c_crop)(tvF.rotate(img, angle))  # rotate and crop to valid data

    if target is None:  # for augmenting unpaired images
        transformer = trf.Compose([trf.RandomCrop(final_crop), trf.RandomHorizontalFlip(), trf.RandomVerticalFlip()])
        new_source_t = transformer(temp_source)
        new_source = conversions(new_source_t, type(image))
        return new_source

    else:  # for augmenting image pairs with the same transformations
        if type(target) != torch.Tensor:
            tgt = tvF.to_tensor(target)
        else:
            tgt = target
        temp_target = trf.CenterCrop(c_crop)(tvF.rotate(tgt, angle))  # rotate and crop to valid data
        c_top = random.randint(0, (c_crop - final_crop))
        c_left = random.randint(0, (c_crop - final_crop))
        flips = random.choice(['h', 'v', 'both', 'none'])

        # flips and crops
        if flips == 'h':
            new_source_t = tvF.crop(tvF.hflip(temp_source), c_top, c_left, final_crop, final_crop)
            new_target_t = tvF.crop(tvF.hflip(temp_target), c_top, c_left, final_crop, final_crop)
        elif flips == 'v':
            new_source_t = tvF.crop(tvF.vflip(temp_source), c_top, c_left, final_crop, final_crop)
            new_target_t = tvF.crop(tvF.vflip(temp_target), c_top, c_left, final_crop, final_crop)
        elif flips == 'both':
            new_source_t = tvF.crop(tvF.hflip(tvF.vflip(temp_source)), c_top, c_left, final_crop, final_crop)
            new_target_t = tvF.crop(tvF.hflip(tvF.vflip(temp_target)), c_top, c_left, final_crop, final_crop)
        else:  # flips == 'none'
            new_source_t = tvF.crop(temp_source, c_top, c_left, final_crop, final_crop)
            new_target_t = tvF.crop(temp_target, c_top, c_left, final_crop, final_crop)

        # convert back to original format
        new_source = conversions(new_source_t, type(image))
        new_target = conversions(new_target_t, type(target))

        return new_source, new_target


def augment(in_path: str, out_path: str, total_imgs: int, min_px=None, max_angle=270):
    """ Augments a set of independent images (unpaired)."""
    if not os.path.isdir(in_path):
        raise NotADirectoryError("Input path is not a directory!")
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    supported = ['.png', '.jpg', '.jpeg', '.xyz']

    def single_aug(filename):
        name, ext = os.path.splitext(filename)
        filepath = os.path.join(in_path, filename)
        if ext.lower() in ['.png', '.jpg', '.jpeg']:  # image extensions must be in this set, other items are skipped
            with Image.open(filepath).convert('RGB') as im:
                im.load()
        elif ext.lower() in ['.xyz']:
            im = xyz_to_zfield(filepath, return3d=True)
        else:
            return None
        transformed_image = random_trf(im, min_dim=min_px, max_angle=max_angle)
        save_name = name + str(index) + ext
        save_path = os.path.join(out_path, save_name)  # image name with index
        if ext.lower() in ['.png', '.jpg', '.jpeg']:
            transformed_image.save(save_path)
        elif ext.lower() in ['.xyz']:
            arr3d_to_xyz(transformed_image, save_path)
        return 1

    pbar = tqdm(total=total_imgs, unit=' files', desc='Augmenting data', leave=True)  # progress bar
    index = 0
    all_imgs = [f for f in os.listdir(in_path) if os.path.splitext(f)[-1].lower() in supported]

    pool = Pool(cpu_count())
    while index < total_imgs:
        if index + len(all_imgs) > total_imgs:
            aug_list = random.sample(all_imgs, (total_imgs - index))
        else:
            aug_list = all_imgs

        recorded = 0
        results = pool.amap(single_aug, aug_list)
        while not results.ready():
            res = results._value
            to_update = sum(filter(None, res)) - recorded
            pbar.update(to_update)
            recorded += to_update
        j = len(aug_list) - recorded
        pbar.update(j)
        index += len(aug_list)
        if index >= total_imgs:
            break
    pbar.close()
    pool.close()
    pool.join()
    pool.clear()
    print("Done!")


def augment_pairs(source_path_in: str, source_path_out: str, target_path_in: str, total_imgs: int, min_px=None, max_angle=270):
    if not (os.path.isdir(source_path_in) and os.path.isdir(target_path_in)):
        raise NotADirectoryError("One of your input paths is not a directory!")
    if not os.path.isdir(source_path_out):
        os.mkdir(source_path_out)
    target_path_out = os.path.join(source_path_out, "targets")
    if not os.path.isdir(target_path_out):
        os.mkdir(target_path_out)
    supported = ['.png', '.jpg', '.jpeg', '.xyz']

    def single_aug(source_file, target_file, idx):
        s_name, ext = os.path.splitext(source_file)
        t_name = os.path.splitext(target_file)[0]
        s_path = os.path.join(source_path_in, source_file)
        t_path = os.path.join(target_path_in, target_file)

        if ext.lower() in ['.png', '.jpg', '.jpeg']:  # image extensions must be in this set, other items are skipped
            with Image.open(s_path).convert('RGB') as source:  # load source image
                source.load()
            with Image.open(t_path).convert('RGB') as target:  # load target image
                target.load()
        elif ext.lower() in ['.xyz']:
            source = xyz_to_zfield(s_path, return3d=True)
            target = xyz_to_zfield(t_path, return3d=True)
        else:
            return None

        transformed_source, transformed_target = random_trf(source, min_px, target, max_angle=max_angle)

        source_save_name = s_name + str(idx) + ext
        target_save_name = t_name + str(idx) + ext
        source_save_path = os.path.join(source_path_out, source_save_name)  # image name with index
        target_save_path = os.path.join(target_path_out, target_save_name)  # image name with "target_" + index

        if ext.lower() in ['.png', '.jpg', '.jpeg']:
            transformed_source.save(source_save_path)
            transformed_target.save(target_save_path)
        elif ext.lower() in ['.xyz']:
            arr3d_to_xyz(transformed_source, source_save_path)
            arr3d_to_xyz(transformed_target, target_save_path)
        return 1

    pbar = tqdm(total=total_imgs, unit=' files', desc='Augmenting data', leave=True)  # progress bar
    index = 0
    all_sources = np.array(sorted([s for s in os.listdir(source_path_in) if os.path.splitext(s)[-1].lower() in supported]))
    all_targets = np.array(sorted([t for t in os.listdir(target_path_in) if os.path.splitext(t)[-1].lower() in supported]))

    pool = Pool(cpu_count())
    while index < total_imgs:
        if index + len(all_sources) > total_imgs:
            ids = random.sample(range(len(all_sources)), (total_imgs - index))
            aug_source_list = all_sources[ids]
            aug_target_list = all_targets[ids]
        else:
            aug_source_list = all_sources
            aug_target_list = all_targets

        recorded = 0
        results = pool.amap(single_aug, aug_source_list, aug_target_list, np.arange(index, index + len(aug_source_list)))
        while not results.ready():
            res = results._value
            to_update = sum(filter(None, res)) - recorded
            pbar.update(to_update)
            recorded += to_update
        j = len(aug_source_list) - recorded
        pbar.update(j)
        index += len(aug_source_list)
        if index >= total_imgs:
            break
    pbar.close()
    pool.close()
    pool.join()
    pool.clear()
    print("Done!")


def split(root_dir: str, ratio=0.8):
    """ Splits a set of images into "train" and "valid" subdirectories.

    """
    all_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
    print("Found {} total images...".format(len(all_files)))
    num_train = int(ratio * len(all_files))
    random.shuffle(all_files)
    train = all_files[:num_train]
    valid = all_files[num_train:]

    train_dir = os.path.join(root_dir, "train")
    valid_dir = os.path.join(root_dir, "valid")
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    if not os.path.isdir(valid_dir):
        os.mkdir(valid_dir)

    for img in train:
        file = os.path.join(root_dir, img)
        shutil.move(file, os.path.join(train_dir, img))
    for img in valid:
        file = os.path.join(root_dir, img)
        shutil.move(file, os.path.join(valid_dir, img))

    print("Final split: {} files in train, {} files in valid.".format(len(train), len(valid)))


def get_test(test_path_in: str, test_path_out: str, num: int, target_path_in=None):
    if not os.path.isdir(test_path_out):
        os.mkdir(test_path_out)
    all_test_files = [f for f in os.listdir(test_path_in) if os.path.isfile(os.path.join(test_path_in, f))]
    all_test_files.sort()
    print("Selecting {} test images...".format(num))
    test_files = random.sample(all_test_files, num)
    for f in test_files:
        filepath = os.path.join(test_path_in, f)
        shutil.copy(filepath, test_path_out)
    if target_path_in is not None:
        all_target_files = [f for f in os.listdir(target_path_in) if os.path.isfile(os.path.join(target_path_in, f))]
        all_target_files.sort()

        if all_test_files == all_target_files:
            target_list = test_files
        elif all_test_files == ['target_' + f for f in all_target_files]:  # 'target_' in all_test_files[0] and 'target_' not in all_target_files[0]:
            target_list = [f.replace('target_', '') for f in test_files]
        elif all_target_files == ['target_' + f for f in all_test_files]:
            target_list = ['target_' + f for f in test_files]
        else:
            raise Exception("Something in the filenames is unexpected. Check for non-matching files.")

        target_path_out = os.path.join(test_path_out, "targets")
        if not os.path.isdir(target_path_out):
            os.mkdir(target_path_out)
        for f in target_list:
            filepath = os.path.join(target_path_in, f)
            shutil.copy(filepath, target_path_out)
    print("Test images saved in: {}".format(test_path_out))


def batch_rename(root_dir, location, add_string, save_dir=None, to_replace=''):
    files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
    if location not in ['first', 'last', 'ext', 'replace']:
        raise ValueError("Mode must be one of: 'first', 'last', or 'ext'. You tried mode {}".format(location))
    if save_dir is None:
        save_dir = root_dir
    for fname in tqdm(files, unit=' file', desc='Renaming files'):
        if location == 'first':
            renamed = add_string + fname
        elif location == 'last':
            renamed = fname + add_string
        elif location == 'replace':
            assert to_replace != '', "The string you're replacing can't be empty!"
            renamed = fname.replace(to_replace, add_string)
        else:  # mode == 'ext':     # default
            renamed = os.path.splitext(fname)[0] + add_string
        old_path = os.path.join(root_dir, fname)
        new_path = os.path.join(save_dir, renamed)
        if not os.path.isfile(new_path):
            shutil.copy(old_path, new_path)
        if root_dir == save_dir:
            os.remove(old_path)

# if __name__ == '__main__':
#     sleep(2)
# ---------------------------------------------------------------- Data Augmenting ----------------------------------------------------------------
# Augmenting data
# source_in_dir = "C:/Users/eva_n/OneDrive - The University of Texas at Austin/SANDIA PHD RESEARCH/Ryan-AFM-Data/Combined-HS20MG-256/z-only-conversions"
# source_out_dir = "C:/Users/eva_n/OneDrive - The University of Texas at Austin/PyCharm Projects/emn-n2n-pytorch/hs20mg_z0nly_data"
# target_in_dir = "C:/Users/eva_n/OneDrive - The University of Texas at Austin/SANDIA PHD RESEARCH/Ryan-AFM-Data/Combined-HS20MG-256/z-only-extra-processed"
#
# number = 1200
# px = 256
# m_angle = 300

# augment(source_in_dir, source_out_dir, number, min_px=px, max_angle=m_angle)
# augment_pairs(source_in_dir, source_out_dir, target_in_dir, number, min_px=px, max_angle=m_angle)

# Splitting data
# split_ratio = 0.8
# split(source_out_dir, split_ratio)

# Getting test images
# nt = 7
# test_out_dir = os.path.join(source_out_dir, "test")
# get_test(source_in_dir, test_out_dir, nt, target_in_dir)
# -------------------------------------------------------------------------------------------------------------------------------------------------
# Dropping x&y from xyz data
# if not os.path.isdir(source_out_dir):
#     os.mkdir(source_out_dir)
# with os.scandir(source_in_dir) as folder:
#     for file in folder:
#         name = os.path.splitext(file.name)[0]
#         save_path = os.path.join(source_out_dir, (name + '.csv'))
#         z = xyz_to_zfield(file.path)
#         np.savetxt(save_path, z, delimiter=',')
# -------------------------------------------------------------------------------------------------------------------------------------------------
# Renaming files
# mode = 'ext'
# new_string = '.txt'
# batch_rename(target_in_dir, mode, new_string)
