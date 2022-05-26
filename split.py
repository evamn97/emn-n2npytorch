""" Splits a dataset directory into training and validation sets based on a given ratio. """
import os
import random
import shutil

ratio = 0.8
root_dir = "hs_20mg_train_data/"
all_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
print(len(all_files), " total files")
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

print(len(train), " files in train")
print(len(valid), " files in valid")
