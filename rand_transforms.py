import numpy as np
import torch as t
import torchvision.io as tv
import torchvision.transforms as trf
import PIL.Image as Image
import os

crop_size = 128  # image square crop size
crop_trf = trf.RandomCrop(crop_size)
# transform image with rotation up to indicated degrees, a horizontal flip, and a vertical flip. Each transform has a p chance of occurring.
rand_trf = trf.RandomApply([trf.RandomRotation(70),
                            trf.RandomHorizontalFlip(),
                            trf.RandomVerticalFlip()], p=0.8)

path = "new-png-conversions"  # relative image folder path
total_images = 1000  # total number of images to create

if os.path.isdir(path):
    save_folder = os.path.join(path, 'new_ml_data')  # saves into folder called "ml_data"
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    index = 0
    while index < total_images:
        with os.scandir(path) as folder:
            for item in folder:
                name, ext = os.path.splitext(item)
                if ext.lower() not in {'.png'}:  # image extensions must be in this set, other items are skipped
                    continue
                image_path = item.path
                torch_image = tv.read_image(image_path,
                                            mode=tv.ImageReadMode.RGB)
                transformed_image = rand_trf(torch_image)
                # final_image = crop_trf(transformed_image)
                final_image = transformed_image  # for no cropping
                image_name = str(index) + '_calib_HS20MG.png'
                save_path = os.path.join(save_folder, image_name)  # image name with index
                tv.write_png(final_image, save_path,
                             compression_level=0)
                print("Saved image", image_name)
                index += 1
                if index > total_images:
                    break
