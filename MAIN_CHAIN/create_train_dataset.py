import os
import random
import shutil
import tqdm as tqdm 

def split_and_copy_images(source_path, train_path, test_path, train_ratio=0.05):
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    image_files = os.listdir(source_path)
    print(len(image_files))
    random.shuffle(image_files)

    num_train = int(len(image_files) * train_ratio)
    train_files = image_files[:num_train]
    test_files = image_files[num_train:]

    for filename in tqdm.tqdm(train_files, total=len(train_files), desc= "Getting Training Files:"):
        source_file = os.path.join(source_path, filename)
        dest_file = os.path.join(train_path, filename)
        shutil.copy(source_file, dest_file)

    for filename in tqdm.tqdm(test_files, total=len(test_files), desc= "Getting Testing Files:"):
        source_file = os.path.join(source_path, filename)
        dest_file = os.path.join(test_path, filename)
        shutil.copy(source_file, dest_file)

if __name__ == "__main__":
    source_path = "/mayo_atlas/atlas/colorectal_cancer/mayo_colon_wsi"
    train_path = "/mayo_atlas/home/m296984/MAIN_CHAIN_CRC_RESULTS/train_liver_data"
    test_path = "/mayo_atlas/home/m296984/MAIN_CHAIN_CRC_RESULTS/test_live_data"

    split_and_copy_images(source_path, train_path, test_path)