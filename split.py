""" Splits a dataset directory into training and validation sets based on a given ratio. """
import os
import random
import shutil

ratio = 0.8
root_dir = "new_ml_data"
all_files = os.listdir(root_dir)
print(len(all_files), " total files")
num_train = int(ratio * len(all_files))
random.shuffle(all_files)
train = all_files[:num_train]
valid = all_files[num_train:]

for img in train:
    shutil.move(os.path.join(root_dir, img), "train\\" + img)

for img in valid:
    shutil.move(os.path.join(root_dir, img), "valid\\" + img)

print(len(train), " files in train")
print(len(valid), " files in valid")
