import PIL.Image
import numpy as np
import random
import torchvision.transforms as trf
from torchvision.transforms import functional as TVF
import PIL.Image as Image
import os
from tqdm import tqdm
import shutil


def highest_pow_2(n):
    p = int(np.log2(n))
    res = int(pow(2, p))
    return res


def random_trf(image: PIL.Image.Image, min_dim=None, target=None, max_angle=270):
    """ Applies random transformations to a single image or pair for data augmentation.
    :param image: Independent image, or source image if using pairs.
    :param min_dim: Minimum dimension of input image(s). For transforming sets of differently sized images, locks size of output images.
    :param target: Corresponding target in an image pair.
    :param max_angle: maximum angle range within which to rotate the image.
    """
    P = image.size[0]
    if min_dim is None:
        min_dim = P
    rng = np.random.default_rng()
    if not max_angle <= 0:
        angle = rng.integers(0, max_angle)  # get random angle in degrees
    else:
        angle = 0
    rad_angle = np.radians(angle)
    c_crop = int(P / (np.abs(np.sin(rad_angle)) + np.abs(np.cos(rad_angle))))  # get bbox size based on rotation
    min_crop = int(min_dim / (2 * np.cos(np.pi / 4)))  # get smallest bbox
    final_crop = highest_pow_2(min_crop)  # must be power of 2
    temp_source = trf.CenterCrop(c_crop)(image.rotate(angle))  # rotate and crop to valid data

    if target is None:  # for augmenting unpaired images
        transformer = trf.Compose([trf.RandomCrop(final_crop), trf.RandomHorizontalFlip(), trf.RandomVerticalFlip()])
        new_source = transformer(temp_source)
        return new_source

    else:  # for augmenting image pairs with the same transformations
        temp_target = trf.CenterCrop(c_crop)(target.rotate(angle))  # rotate and crop to valid data
        c_top = random.randint(0, (c_crop - final_crop))
        c_left = random.randint(0, (c_crop - final_crop))
        flips = random.choice(['h', 'v', 'both', 'none'])

        # flips and crops
        if flips == 'h':
            new_source = TVF.crop(TVF.hflip(temp_source), c_top, c_left, final_crop, final_crop)
            new_target = TVF.crop(TVF.hflip(temp_target), c_top, c_left, final_crop, final_crop)
        elif flips == 'v':
            new_source = TVF.crop(TVF.vflip(temp_source), c_top, c_left, final_crop, final_crop)
            new_target = TVF.crop(TVF.vflip(temp_target), c_top, c_left, final_crop, final_crop)
        elif flips == 'both':
            new_source = TVF.crop(TVF.hflip(TVF.vflip(temp_source)), c_top, c_left, final_crop, final_crop)
            new_target = TVF.crop(TVF.hflip(TVF.vflip(temp_target)), c_top, c_left, final_crop, final_crop)
        else:  # flips == 'none'
            new_source = TVF.crop(temp_source, c_top, c_left, final_crop, final_crop)
            new_target = TVF.crop(temp_target, c_top, c_left, final_crop, final_crop)

        return new_source, new_target


def augment(in_path: str, out_path: str, total_imgs: int, min_px=None, max_angle=270):
    """ Augments a set of independent images (unpaired).

    """
    if os.path.isdir(in_path):
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
        pbar = tqdm(total=total_imgs, unit='file', desc='Augmenting data', leave=True)  # progress bar
        index = 0
        while index < total_imgs:
            with os.scandir(in_path) as folder:
                for item in folder:
                    name, ext = os.path.splitext(item.name)
                    if ext.lower() not in {'.png', '.jpg', '.jpeg'}:  # image extensions must be in this set, other items are skipped
                        continue
                    with Image.open(item.path).convert('RGB') as im:
                        im.load()
                    if min_px is None:
                        min_px = im.size[0]
                    transformed_image = random_trf(im, min_dim=min_px, max_angle=max_angle)
                    save_name = name + str(index) + ext
                    save_path = os.path.join(out_path, save_name)  # image name with index
                    transformed_image.save(save_path)
                    pbar.update(1)
                    index += 1
                    if index == total_imgs:
                        break
        pbar.close()
    else:
        raise AssertionError("Input path is not a directory!")


def augment_pairs(source_path_in: str, source_path_out: str, target_path_in: str, total_imgs: int, min_px=None, max_angle=270):
    assert (os.path.isdir(source_path_in) and os.path.isdir(target_path_in)), "One of your input paths is not a directory!"
    if not os.path.isdir(source_path_out):
        os.mkdir(source_path_out)
    target_path_out = os.path.join(source_path_out, "targets")
    if not os.path.isdir(target_path_out):
        os.mkdir(target_path_out)

    pbar = tqdm(total=total_imgs, unit='file', desc='Augmenting data', leave=True)  # progress bar
    index = 0
    while index < total_imgs:
        s_folder = sorted(os.scandir(source_path_in), key=lambda e: e.name)
        t_folder = sorted(os.scandir(target_path_in), key=lambda e: e.name)
        for (s, t) in zip(s_folder, t_folder):
            s_name, ext = os.path.splitext(s.name)
            t_name = os.path.splitext(t.name)[0]
            if ext.lower() not in {'.png'}:  # image extensions must be in this set, other items are skipped
                continue
            with Image.open(s.path).convert('RGB') as source:  # load source image
                source.load()
            with Image.open(t.path).convert('RGB') as target:  # load target image
                target.load()

            transformed_source, transformed_target = random_trf(source, min_px, target, max_angle=max_angle)

            source_save_name = s_name + str(index) + ext
            target_save_name = t_name + str(index) + ext
            source_save_path = os.path.join(source_path_out, source_save_name)  # image name with index
            target_save_path = os.path.join(target_path_out, target_save_name)  # image name with "target_" + index

            transformed_source.save(source_save_path)  # gives attribute error here but this works for PIL images
            transformed_target.save(target_save_path)

            pbar.update(1)
            index += 1
            if index == total_imgs:
                break
    pbar.close()


def split(root_dir: str, ratio=0.8):
    """ Splits a set of images into "train" and "valid" subdirectories.

    """
    all_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
    print(len(all_files), " total images")
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
    all_files = [f for f in os.listdir(test_path_in) if os.path.isfile(os.path.join(test_path_in, f))]
    print("Selecting {} test images...".format(num))
    test_files = random.sample(all_files, num)
    for f in test_files:
        filepath = os.path.join(test_path_in, f)
        shutil.copy(filepath, test_path_out)
    if target_path_in is not None:
        target_list = ['target_' + f for f in test_files]
        target_path_out = os.path.join(test_path_out, "targets")
        if not os.path.isdir(target_path_out):
            os.mkdir(target_path_out)
        for f in target_list:
            filepath = os.path.join(target_path_in, f)
            shutil.copy(filepath, target_path_out)
    print("Test images saved in: {}".format(test_path_out))


if __name__ == '__main__':
    # Augmenting data
    source_in_dir = "C:/Users/eva_n/OneDrive - The University of Texas at Austin/SANDIA PHD RESEARCH/Ryan-AFM-Data/Combined-HS20MG-256/planelevel-bw-png-files"
    source_out_dir = "C:/Users/eva_n/OneDrive - The University of Texas at Austin/PyCharm Projects/emn-n2n-pytorch/planelevel_bw_hs20mg_data"
    target_in_dir = "C:/Users/eva_n/OneDrive - The University of Texas at Austin/SANDIA PHD RESEARCH/Ryan-AFM-Data/Combined-HS20MG-256/bw-processed-pngs"

    number = 1200
    px = 256
    max_angle = 0

    # augment(source_in_dir, source_out_dir, number, min_px=px, max_angle=max_angle)
    augment_pairs(source_in_dir, source_out_dir, target_in_dir, number, min_px=px, max_angle=max_angle)

    # Splitting data
    split_ratio = 0.8
    split(source_out_dir, split_ratio)

    # Getting test images
    nt = 7
    test_out_dir = os.path.join(source_out_dir, "test")
    get_test(source_in_dir, test_out_dir, nt, target_in_dir)
