import os
import random
import shutil
from typing import Union

import PIL.Image as Image
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as trf
from pathos.helpers import cpu_count
from pathos.pools import ProcessPool as Pool
from torchvision.transforms import functional as tvF
from tqdm import tqdm


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
    """Loads .xyz (3-column) or z-series (1 column) data and converts into a square numpy array."""
    df = pd.read_csv(xyz_filepath, header=None, delimiter='\t', index_col=False)
    P = int(np.sqrt(len(df)))
    try:
        z_arr = df[df.columns[-1]].values.reshape((P, -1))
    except ValueError:
        P = highest_pow_2(P)
        z_arr = df[df.columns[-1]].values.reshape((P, -1))

    if return3d and len(df.columns) == 3:
        x_arr = df[df.columns[0]].values.reshape((P, -1))
        y_arr = df[df.columns[1]].values.reshape((P, -1))
        arr_3d = np.stack((x_arr, y_arr, z_arr), axis=2)
        return arr_3d
    else:
        if return3d:
            print('Return3d was requested but input df does not have 3 columns. Returning 2D array.')
        return z_arr


def arr3d_to_xyz(arr3d, out_path):
    if len(arr3d.shape) == 3:  # 3D array
        xyz_df = pd.DataFrame({'x': arr3d[:, :, 0].flat, 'y': arr3d[:, :, 1].flat, 'z': arr3d[:, :, 2].flat})
        xyz_df.to_csv(out_path, header=False, index=False, sep='\t')
    else:  # 2D array
        z_df = pd.DataFrame({'z': arr3d.flat})
        z_df.to_csv(out_path, header=False, index=False, sep='\t')


def find_target(targets_list: list, source_name: str):
    # first check if source_name is an exact substring of target:
    res = [t for t in targets_list if (source_name in t or t in source_name)]
    if len(res) == 1:
        return res[0]

    # if there's no direct match, try partial prefix matching
    else:
        sub = max([os.path.commonprefix([t, source_name]) for t in targets_list], key=len)
        res = [t for t in targets_list if sub in t]
        if len(res) == 1:
            return res[0]

        # if there's more than one direct match, try once more then raise error
        else:
            # try matching with substring one character longer bc commonprefix usually tries to return multiple results
            #   (e,g,. "n03599486_220_corrupt" returns matches "n03599486_220_clean" *and* "n03599486_220_creepclean" because suffixes "corrupt" and "creepclean" both start with "c")
            res2 = [t for t in targets_list if (source_name[:len(sub)+1] in t or source_name[:len(sub)+2] in t)]
            if len(res2) == 1:
                return res2[0]
            # if that doesn't work, raise error with problematic substring and source name for reference
            else:
                raise ValueError(
                    f'Expected single target for {source_name}, got {len(res)} from substrings {sub} and {source_name[:len(sub)+2]}. ',
                    f'Check file naming: source and target are matched by prefix (assumes difference is a suffix, e.g., "_corrupted"). See src.data_prep.find_target() for details.')


def conversions(f_in, new_type):
    supported = [Image.Image, torch.Tensor, np.ndarray]
    if type(f_in) not in supported:
        raise TypeError("Unsupported input type.")
    if new_type not in supported:
        raise TypeError("Unsupported conversion type.")
    assert type(f_in) != new_type, "Input type matches conversion type."

    if new_type == Image.Image:
        if type(f_in) == torch.Tensor:
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


def random_trf(image: Union[Image.Image, torch.Tensor, np.ndarray], min_dim=None, target=None, corner_priority=False, max_angle=270.0):
    """ Applies random transformations to a single image or pair for data augmentation.
    :param image: Independent image, or source image if using pairs.
    :param min_dim: Minimum dimension of input image(s). For transforming sets of differently sized images, locks size of output images.
    :param target: Corresponding target in an image pair.
    :param corner_priority: if True, sets rotation angle to zero and forces cropping bbox on images to enclose corners of the original image.
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
    if not max_angle <= 0 and not corner_priority:
        angle = rng.uniform(0, max_angle)  # get random angle in degrees
    else:
        angle = 0
    rad_angle = np.radians(angle)
    c_crop = int(P / (np.abs(np.sin(rad_angle)) + np.abs(np.cos(rad_angle))))  # get bbox size based on rotation
    min_crop = int(min_dim / (2 * np.cos(np.pi / 4)))  # get smallest bbox based on a 45 deg rotation angle
    final_crop = highest_pow_2(min_crop)  # must be power of 2 (usually one power of 2 smaller than min_dim)

    if corner_priority:
        c_top = random.choice([0, (c_crop - final_crop)])
        c_left = random.choice([0, (c_crop - final_crop)])
    else:
        c_top = random.randint(0, (c_crop - final_crop))
        c_left = random.randint(0, (c_crop - final_crop))

    temp_source = trf.CenterCrop(c_crop)(tvF.rotate(img, angle))  # rotate and crop to valid data

    if target is None:  # for augmenting unpaired images
        if corner_priority:
            transformer = trf.Compose([trf.RandomHorizontalFlip(), trf.RandomVerticalFlip()])
            new_source_t = tvF.crop(transformer(temp_source), c_top, c_left, final_crop, final_crop)
        else:
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

        # flips and crops
        flips = random.choice(['h', 'v', 'both', 'none'])
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


def augment(in_path: str, out_path: str, total_imgs: int, min_px=None, max_angle=270, corners=True):
    """ Augments a set of independent images (unpaired)."""
    if not os.path.isdir(in_path):
        raise NotADirectoryError("Input path is not a directory!")
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    supported = ['.png', '.jpg', '.jpeg', '.xyz', '.txt', '.csv']

    def single_aug(filename, idx):
        name, ext = os.path.splitext(filename)
        filepath = os.path.join(in_path, filename)
        if ext.lower() in ['.png', '.jpg', '.jpeg']:  # image extensions must be in this set, other items are skipped
            with Image.open(filepath) as im:
                im.load()
        elif ext.lower() in ['.xyz']:
            im = xyz_to_zfield(filepath, return3d=True)
        elif ext.lower() in ['.txt', '.csv']:
            im = xyz_to_zfield(filepath)
        else:
            return None

        corner_priority = False
        if corners:
            corner_idx = random.choice(mini_index)
            if idx == corner_idx:
                corner_priority = True

        transformed_image = random_trf(im, min_dim=min_px, max_angle=max_angle, corner_priority=corner_priority)
        save_name = name + str(index) + str(idx) + ext
        save_path = os.path.join(out_path, save_name)  # image name with index
        if ext.lower() in ['.png', '.jpg', '.jpeg']:
            transformed_image.save(save_path)
        elif ext.lower() in ['.xyz', '.txt', '.csv']:
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

        mini_index = np.arange(0, len(aug_list))
        recorded = 0
        results = pool.amap(single_aug, aug_list, mini_index)
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


def augment_pairs(source_path_in: str, source_path_out: str, target_path_in: str, total_imgs: int, min_px=None, max_angle=270, corners=True):
    if not (os.path.isdir(source_path_in) and os.path.isdir(target_path_in)):
        raise NotADirectoryError("One of your input paths is not a directory!")
    if not os.path.isdir(source_path_out):
        os.mkdir(source_path_out)
    target_path_out = os.path.join(source_path_out, "targets")
    if not os.path.isdir(target_path_out):
        os.mkdir(target_path_out)
    supported = ['.png', '.jpg', '.jpeg', '.xyz', '.txt', '.csv']

    def single_aug(source_file, target_file, idx):
        s_name, ext = os.path.splitext(source_file)
        t_name = os.path.splitext(target_file)[0]
        s_path = os.path.join(source_path_in, source_file)
        t_path = os.path.join(target_path_in, target_file)

        if ext.lower() in ['.png', '.jpg', '.jpeg']:  # image extensions must be in this set, other items are skipped
            with Image.open(s_path) as source:  # load source image
                source.load()
            with Image.open(t_path) as target:  # load target image
                target.load()
        elif ext.lower() in ['.xyz']:
            source = xyz_to_zfield(s_path, return3d=True)
            target = xyz_to_zfield(t_path, return3d=True)
        elif ext.lower() in ['.txt', '.csv']:
            source = xyz_to_zfield(s_path)
            target = xyz_to_zfield(t_path)
        else:
            return None

        corner_priority = False
        if corners:
            corner_idx = random.choice(mini_index)
            if idx == corner_idx:
                corner_priority = True

        transformed_source, transformed_target = random_trf(source, min_px, target, max_angle=max_angle, corner_priority=corner_priority)

        source_save_name = s_name + str(index) + str(idx) + ext  # ex: HS-20MG00.ext, HS-20MG01.ext
        target_save_name = t_name + str(index) + str(idx) + ext
        source_save_path = os.path.join(source_path_out, source_save_name)  # image name with index
        target_save_path = os.path.join(target_path_out, target_save_name)  # image name with "target_" + index

        if ext.lower() in ['.png', '.jpg', '.jpeg']:
            transformed_source.save(source_save_path)
            transformed_target.save(target_save_path)
        elif ext.lower() in ['.xyz', '.txt', '.csv']:
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

        mini_index = np.arange(0, len(aug_source_list))

        recorded = 0
        results = pool.amap(single_aug, aug_source_list, aug_target_list, mini_index)
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

        target_path_out = os.path.join(test_path_out, "targets")
        if not os.path.isdir(target_path_out):
            os.mkdir(target_path_out)
        for f in test_files:
            filepath = os.path.join(target_path_in, find_target(all_target_files, f))  # get target path
            shutil.copy(filepath, target_path_out)
    print("Test images saved in: {}".format(test_path_out))


def batch_rename(root_dir, location, add_string, save_dir=None, to_replace=''):
    """ Batch file renaming for all items in a given directory. """
    files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]  # for files only
    if location not in ['first', 'last', 'replace', 'ext']:
        raise ValueError("Mode must be one of: 'first', 'last', 'replace', or 'ext'. You tried mode {}".format(location))
    if save_dir is None:
        save_dir = root_dir
    for fname in tqdm(files, unit=' file', desc='Renaming files'):
        if location == 'first':
            renamed = add_string + fname
        elif location == 'last':
            renamed = os.path.splitext(fname)[0] + add_string + os.path.splitext(fname)[1]
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
    print("Done!")
