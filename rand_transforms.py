import PIL.Image
import numpy as np
import random
import torchvision.transforms as trf
from torchvision.transforms import functional as TVF
import PIL.Image as Image
import os
from tqdm import tqdm


def random_trf(image: PIL.Image.Image, min_dim=None, target=None):
    """ Applies random transformations to a single image or pair for data augmentation.
    :param image: Independent image, or source image if using pairs.
    :param min_dim: Minimum dimension of input image(s). For transforming sets of differently sized images, locks size of output images.
    :param target: Corresponding target in an image pair.
    """
    P = image.size[0]
    if min_dim is None:
        min_dim = P
    rng = np.random.default_rng()
    angle = rng.integers(0, 270)  # get random angle in degrees
    rad_angle = np.radians(angle)
    c_crop = int(P / (np.abs(np.sin(rad_angle)) + np.abs(np.cos(rad_angle))))  # get bbox size based on rotation
    min_crop = int(min_dim / (2 * np.cos(np.pi / 4)))  # get smallest bbox
    temp_source = trf.CenterCrop(c_crop)(image.rotate(angle))  # rotate and crop to valid data

    if target is None:  # for augmenting unpaired images
        transformer = trf.Compose([trf.RandomCrop(min_crop), trf.RandomHorizontalFlip(), trf.RandomVerticalFlip()])
        new_source = transformer(temp_source)
        return new_source

    else:  # for augmenting image pairs with the same transformations
        temp_target = trf.CenterCrop(c_crop)(target.rotate(angle))  # rotate and crop to valid data
        c_top = random.randint(0, (c_crop - min_crop))
        c_left = random.randint(0, (c_crop - min_crop))
        flips = random.choice(['h', 'v', 'both', 'none'])

        # flips and crops
        if flips == 'h':
            new_source = TVF.crop(TVF.hflip(temp_source), c_top, c_left, min_crop, min_crop)
            new_target = TVF.crop(TVF.hflip(temp_target), c_top, c_left, min_crop, min_crop)
        elif flips == 'v':
            new_source = TVF.crop(TVF.vflip(temp_source), c_top, c_left, min_crop, min_crop)
            new_target = TVF.crop(TVF.vflip(temp_target), c_top, c_left, min_crop, min_crop)
        elif flips == 'both':
            new_source = TVF.crop(TVF.hflip(TVF.vflip(temp_source)), c_top, c_left, min_crop, min_crop)
            new_target = TVF.crop(TVF.hflip(TVF.vflip(temp_target)), c_top, c_left, min_crop, min_crop)
        else:  # flips == 'none'
            new_source = TVF.crop(temp_source, c_top, c_left, min_crop, min_crop)
            new_target = TVF.crop(temp_target, c_top, c_left, min_crop, min_crop)

        return new_source, new_target


def augment(in_path, out_path, total_imgs, min_px=None):
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
                    if ext.lower() not in {'.png'}:  # image extensions must be in this set, other items are skipped
                        continue
                    with Image.open(item.path) as im:
                        im.convert('RGB')
                        im.load()
                    if min_px is None:
                        min_px = im.size[0]
                    transformed_image = random_trf(im, min_px)
                    save_name = name + str(index) + ext
                    save_path = os.path.join(out_path, save_name)  # image name with index
                    transformed_image.save(save_path)
                    pbar.update(1)
                    index += 1
                    if index > total_imgs:
                        break
        pbar.close()
    else:
        raise AssertionError("Input path is not a directory!")


def augment_pairs(source_path_in, source_path_out, target_path_in, target_path_out, total_imgs, min_px):
    assert (os.path.isdir(source_path_in) and os.path.isdir(target_path_in)), "One of your input paths is not a directory!"
    if not os.path.isdir(source_path_out):
        os.mkdir(source_path_out)
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
            with Image.open(s.path) as source:  # load source image
                source.convert('RGB')
                source.load()
            with Image.open(t.path) as target:  # load target image
                target.convert('RGB')
                target.load()
            if min_px is None:  # get size from source if not specified
                min_px = s.size[0]

            transformed_source, transformed_target = random_trf(source, min_px, target)

            source_save_name = s_name + str(index) + ext
            target_save_name = t_name + str(index) + ext
            source_save_path = os.path.join(source_path_out, source_save_name)  # image name with index
            target_save_path = os.path.join(target_path_out, target_save_name)  # image name with index + "_target"

            transformed_source.save(source_save_path)  # gives attribute error here but this works for PIL images
            transformed_target.save(target_save_path)

            pbar.update(1)
            index += 1
            if index > total_imgs:
                break
    pbar.close()


if __name__ == '__main__':
    input_dir = "C:/Users/eva_n/OneDrive - The University of Texas at Austin/SANDIA PHD RESEARCH/Ryan-AFM-Data/Combined-HS20MG-256/processed_pngs"
    aug_dir = "C:/Users/eva_n/OneDrive - The University of Texas at Austin/PyCharm Projects/emn-n2n-pytorch/hs_20mg_train_data"
    # augment(input_dir, aug_dir, 256, 1200)

    source_in_dir = "C:/Users/eva_n/OneDrive - The University of Texas at Austin/SANDIA PHD RESEARCH/Ryan-AFM-Data/Combined-HS20MG-256/png_files"
    target_in_dir = "C:/Users/eva_n/OneDrive - The University of Texas at Austin/SANDIA PHD RESEARCH/Ryan-AFM-Data/Combined-HS20MG-256/processed_pngs"
    source_out_dir = "C:/Users/eva_n/OneDrive - The University of Texas at Austin/PyCharm Projects/emn-n2n-pytorch/hs_20mg_train_data"
    target_out_dir = "C:/Users/eva_n/OneDrive - The University of Texas at Austin/PyCharm Projects/emn-n2n-pytorch/hs_20mg_train_data/targets"
    number = 1200
    px = 256

    augment_pairs(source_in_dir, source_out_dir, target_in_dir, target_out_dir, number, px)
