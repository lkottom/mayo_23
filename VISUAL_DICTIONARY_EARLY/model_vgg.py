import numpy as np
import keras
import os
from custom_dataset import CustomDataset
from keras.applications import VGG16
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Model
from multiprocessing import Pool
import tqdm as tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load VGG-16 with pretrained weights (excluding the top fully-connected layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Remove the final classification layers to get feature embeddings
model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

# Function to preprocess and extract features from an image
def extract_features(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Resize the image to (224, 224)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = keras.applications.vgg16.preprocess_input(img)
    features = model.predict(img, verbose=0)
    return features.flatten()


def process_batch(args):
    batch, num_words = args
    for _ in range(num_words):
        print(batch[1][0])
    
    # return [extract_features(batch[1][0]) for _ in range(len(batch))]

def get_embeddings_VGG_multiprocess(slide_data_path, save_path, num_words, num_workers=4):
    dataset = CustomDataset(data_path=slide_data_path, transform=transforms.ToTensor())
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=num_workers)

    with Pool(processes=num_workers) as pool:
        args = loader, num_words
        embeddings_list = list(pool.imap(process_batch, args))
    
    print(embeddings_list)

    for i, feature_batch in enumerate(embeddings_list):
        feature = feature_batch[0]
        file_name = f"embedding_{i}.pt"
        file_path = os.path.join(save_path, file_name)
        torch.save(feature, file_path)

# For earlier function, not for multiprocessing
# # Example usage:
# image_path = '/mayo_atlas/home/m296984/visual_dictionary_pipeline/train/1000_words_per_slide/NASH_NASH26/image_919_9ba61c68-5ccd-469b-8df8-73e31ab5a9d5.png'
# patch_embedding = extract_features(image_path)
# print(patch_embedding)