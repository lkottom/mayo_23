import pandas as pd
from custom_dataset import CustomDataset
import os
import uuid
from PIL import Image 
import random
import numpy as np
import cv2
import numpy as np
from openslide import OpenSlide
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from multiprocessing import Pool
import tqdm as tqdm




def plot_thumbnail_and_mask(thumbnail, tissue_mask, patch_locations,  mask_hratio, mask_wratio):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the thumbnail
    ax[0].imshow(thumbnail)
    ax[0].set_title('Thumbnail')
    
    # Plot the patches on the thumbnail
    for (x, y) in patch_locations:
        rect = patches.Rectangle((x, y), mask_hratio, mask_wratio, linewidth=1, edgecolor='r', facecolor='none')
        ax[0].add_patch(rect)

    # Plot the tissue mask
    ax[1].imshow(tissue_mask, cmap='gray')
    ax[1].set_title('Tissue Mask')

    plt.savefig('plot.png')  # save the figure
    plt.show()


def RGB2HSD(X):
    eps = np.finfo(float).eps
    X[np.where(X==0.0)] = eps
    
    OD = -np.log(X / 1.0)
    D  = np.mean(OD,3)
    D[np.where(D==0.0)] = eps
    
    cx = OD[:,:,:,0] / (D) - 1.0
    cy = (OD[:,:,:,1]-OD[:,:,:,2]) / (np.sqrt(3.0)*D)
    
    D = np.expand_dims(D,3)
    cx = np.expand_dims(cx,3)
    cy = np.expand_dims(cy,3)
            
    X_HSD = np.concatenate((D,cx,cy),3)
    return X_HSD


def clean_thumbnail(thumbnail):
    thumbnail_arr = np.asarray(thumbnail)
    
    wthumbnail = np.zeros_like(thumbnail_arr)
    wthumbnail[:, :, :] = thumbnail_arr[:, :, :]

    thumbnail_std = np.std(wthumbnail, axis=2)
    wthumbnail[thumbnail_std<5] = (np.ones((1,3), dtype="uint8")*255)
    thumbnail_HSD = RGB2HSD( np.array([wthumbnail.astype('float32')/255.]) )[0]
    kernel = np.ones((30,30),np.float32)/900
    thumbnail_HSD_mean = cv2.filter2D(thumbnail_HSD[:,:,2],-1,kernel)
    wthumbnail[thumbnail_HSD_mean<0.05] = (np.ones((1,3),dtype="uint8")*255)
    return wthumbnail

                
def is_far_enough(new_point, existing_points, min_distance):
    for point in existing_points:
        if np.sqrt((new_point[0] - point[0])**2 + (new_point[1] - point[1])**2) < min_distance:
            return False
    return True


def get_patch_locations(tissue_mask,  mask_hratio, mask_wratio, tissue_threshold):
    # Find contours
    contours, _ = cv2.findContours(tissue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    patch_locations = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= mask_wratio and h >= mask_hratio:
            for i in range(x, x + w - mask_wratio, mask_wratio):
                for j in range(y, y + h - mask_hratio, mask_hratio):
                    tissue_patch = tissue_mask[j:j + mask_hratio, i:i + mask_wratio]
                    # if np.sum(tissue_patch) / (mask_hratio ** 2) > tissue_threshold:
                    if np.count_nonzero(tissue_patch)/tissue_patch.size  >= tissue_threshold:
                        patch_locations.append((i, j))
                
    return patch_locations

def otsu_mask(img, kernel_size=(3, 3)):
    """Segment given thumbnail image using morphological methods for given kernel size"""
    # Change image to gray if it's not already in grayscale
    if len(img.shape) == 3:
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = img.copy()  # If the image is already in grayscale, create a copy

    # Segment image using Otsu's thresholding
    # 0 for black and 255 for white (inverted)
    # Inverting so that background is black, object is white
    _, threshold = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Use opening and closing (cv2 morphological operations) according to kernel value
    # This helps with cleaning noises that don't include tissue parts
    # The bigger the kernel is, the more information will be excluded, so be careful
    kernel = np.ones(kernel_size, np.uint8)
    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    return closing

def extract_patches_from_numpy(patch_list, patch_size=32, num_words=50, min_tissue_percentage=90):
    # Get the dimensions of the tissue image
    random_patches = []
    for image in patch_list:
        image_np = np.array(image)
        image_height, image_width, _ = image_np.shape
        # Generate the tissue mask using Otsu's thresholding (you can replace this with your own mask generation function)
        tissue_mask = otsu_mask(image_np)

        patches_saved = 0
        
        while patches_saved < num_words:
            # Randomly select a location within the tissue image
            x = random.randint(0, image_width - patch_size)
            y = random.randint(0, image_height - patch_size)

            # Extract the patch from the tissue image
            patch = image_np[y:y + patch_size, x:x + patch_size]
            binary_patch = tissue_mask[y:y + patch_size, x:x + patch_size]

            # Check if the patch contains at least min_tissue_percentage tissue
            # tissue_percentage = np.count_nonzero(binary_patch) / (patch_size * patch_size) * 100
            # if tissue_percentage >= min_tissue_percentage:
            patch = Image.fromarray(np.uint8(patch))
            random_patches.append(patch)
            patches_saved += 1
                # if patches_saved >= num_of_words:
                #     break

        # Check if no valid patches were found for this image and skip to the next one
        if patches_saved == 0:
            print(f"No valid patches found for image {image_filename}. Skipping...")
            continue

    return random_patches
            

def process_wsi(wsi_path, 
                number_of_words_per_patch, 
                word_size, 
                output_patch_size=1024, 
                number_of_patches=20, 
                tissue_percent=0.90):
    try:
        wsi_obj = OpenSlide(wsi_path)
        
        wsi_name = Path(wsi_path).stem + ".svs"
        
        thumbnail = wsi_obj.get_thumbnail((1024, 1024))
        cthumbnail = clean_thumbnail(thumbnail)
        tissue_mask = ((cthumbnail.mean(axis=2) != 255) * 255).astype(np.uint8)
        
        w, h = wsi_obj.dimensions
        
        # print(wsi_obj.dimensions)
        if 'openslide.objective-power' in wsi_obj.properties:
            objective_power = int(wsi_obj.properties['openslide.objective-power'])
        else:
            # objective_power = some_default_value  # Replace this with a suitable default value
            pixel_size = wsi_obj.level_downsamples[0]
            objective_power = np.round(np.log2(w / pixel_size) * 100)

        patch_size = (objective_power / 20.) * 1000
        mask_hratio = int((tissue_mask.shape[0] / h) * patch_size)
        mask_wratio = int((tissue_mask.shape[1] / w) * patch_size)
       
        Mask_to_WSI_ratioW = int(w / tissue_mask.shape[1])
        Mask_to_WSI_ratioH = int(h / tissue_mask.shape[0])
        
        patch_locations = get_patch_locations(tissue_mask, mask_hratio, mask_wratio, tissue_percent)

        min_distance = mask_hratio  # Minimum distance between points

        filtered_patch_locations = []
        for (x, y) in patch_locations:
            if is_far_enough((x, y), filtered_patch_locations, min_distance):
                filtered_patch_locations.append((x, y))

        
        scaled_patch_coordinates = []
        for (x, y) in filtered_patch_locations:
            scaled_patch_coordinates.append((int(x * Mask_to_WSI_ratioW), int(y * Mask_to_WSI_ratioH)))
          

        # shuffle the patch locations
        random.shuffle(scaled_patch_coordinates)
        # pick the first n patch locations
        selected = scaled_patch_coordinates[:number_of_patches]
        
        # print(f"Selected {len(selected)} patches in {wsi_name}")
        
        # Iterate through each patch location
        patches_list = []
        for (x, y) in selected:
            # Extract the patch from the WSI
            patch_size_20x = int((objective_power/20.)*output_patch_size)
            patch = wsi_obj.read_region((x, y), 0, (patch_size_20x, patch_size_20x))
            if patch.size[0] != output_patch_size:
                patch = patch.resize((output_patch_size, output_patch_size))
            # Convert the patch to RGB and save it
            patch_rgb = patch.convert('RGB')
            # patch_rgb = np.array(patch.convert('RGB'))
            patches_list.append(patch_rgb)
            
        # Compute the total number of 'words' we want per large patch    
        total_num_of_desired_words = number_of_words_per_patch * number_of_patches
        
        # Compute a new value for how many words we need to get from the patch. If 
        # the length of the selected patches is the same as number_of_patches, this value 
        # will be the same as number_of_words_per_patch. However, if it is less, we 
        # need to make sure we have the same total number of words for each slide. So the 
        # lower the selected patch value, the more words we need to pull from a patch. 
        new_words_needed = total_num_of_desired_words / len(selected)
        
            
        small_patches_list = extract_patches_from_numpy(patches_list, 
                                                        patch_size=word_size, 
                                                        num_words=new_words_needed, 
                                                        min_tissue_percentage=tissue_percent)
        
        # now patch 
        return small_patches_list

    except Exception as e:
        print(f"Error processing {wsi_path}: {e}")
        return False
    
    
if __name__ == "__main__":
    num_words_list = [20]
    patch_size = 256
    
    for num_words in num_words_list:
        keywords = ["test"]
        num_of_words = num_words
        
        for keyword in keywords:
            wsi_dir = f'/mayo_atlas/home/m296984/MAIN_CHAIN_LIVER_RESULTS/testing_data'
            save_dir = f'/mayo_atlas/home/m296984/TEST_20X/{keyword}/{num_of_words}_words_per_slide_{patch_size}'
            os.makedirs(save_dir, exist_ok=True)
            
            words_per_patch = num_of_words/20
            
            # print(f'Processing {keyword} set for {num_words} number of words per slide')
            
            for file in tqdm.tqdm(os.listdir(wsi_dir), total=len(os.listdir(wsi_dir)), desc="Getting Words: "):
                if file.endswith('.svs'):
                    wsi_path = os.path.join(wsi_dir, file)
                    wsi_name = os.path.splitext(os.path.basename(wsi_path))[0]
                    wsi_save_dir = os.path.join(save_dir, wsi_name)
                    image_list = process_wsi(wsi_path, number_of_words_per_patch=words_per_patch, word_size=patch_size, output_patch_size=1024)
                    if (type(image_list)) != bool:
                        os.makedirs(wsi_save_dir, exist_ok=True)
                        for i, image_data in enumerate(image_list):
                            image = image_data
                            image_uuid = str(uuid.uuid4())
                            image_filename = f"image_{i}_{image_uuid}.png"  # Unique filename using uuid
                            image.save(os.path.join(wsi_save_dir, image_filename))
                            

            print("Images saved successfully!")
            