from model_vae_16_dim import Autoencoder
import torch
from torch.utils.data import DataLoader 
from torchvision import transforms
import os
import keras
import tqdm
import numpy as np
from model_vgg import get_embeddings_VGG_multiprocess, extract_features
from custom_dataset import CustomDataset
from keras.applications import VGG16
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATCH_SIZE = 32

def get_embeddings_VAE(model_path, patch_size, slide_data_path, save_path):
    model = Autoencoder(patch_size)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    dataset = CustomDataset(data_path=slide_data_path, transform=transforms.ToTensor())
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=32)
    for i, (photo, _) in enumerate(loader):
        photo = photo.to(device)
        _, embedding, _ = model(photo)
        file_name = f"embedding_{i}.pt"
        file_path = os.path.join(save_path, file_name)
        
        # Save the current embedding tensor to the specified file path
        torch.save(embedding, file_path)

def get_embeddings_VGG(slide_data_path, save_path):
    # get_embeddings_VGG_multiprocess(slide_data_path, save_path, 1000, 4)
    dataset = CustomDataset(data_path=slide_data_path, transform=transforms.ToTensor())
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=32)
    for i, (_, img_path) in enumerate(loader):
        feature = extract_features(img_path[0])
        file_name = f"embedding_{i}.pt"
        file_path = os.path.join(save_path, file_name)
        torch.save(feature, file_path)
   
        
if __name__ == "__main__":
    num_words = 500
    word_size = 32
    model = 'VAE_32x32_16_dim'
    keywords = ["test"]
    for keyword in keywords: 
        MODEL_PATH = '/mayo_atlas/home/m296984/MAIN_CHAIN_LIVER_RESULTS/saved_models/checkpoint_15.pt'
        DATA_PATH = f'/mayo_atlas/home/m296984/visual_dictionary_pipeline/{keyword}/{num_words}_words_per_slide_{word_size}'
        SAVE_PATH = f'/mayo_atlas/home/m296984/visual_dictionary_pipeline/{keyword}/embeddings_{num_words}_{model}'
        os.makedirs(SAVE_PATH, exist_ok=True)
        slide_names = os.listdir(DATA_PATH)
        for slide_name in tqdm.tqdm(slide_names, total=len(slide_names), desc="Saving Embeddings:"):
            slide_save_path = os.path.join(SAVE_PATH, slide_name)
            slide_data_path = os.path.join(DATA_PATH, slide_name)
            if any(os.scandir(slide_data_path)):
                os.makedirs(slide_save_path, exist_ok=True)
                # if model == "VAE_32x32":
                get_embeddings_VAE(MODEL_PATH, PATCH_SIZE, slide_data_path, slide_save_path)
                # elif model == 'VGG':
                #     get_embeddings_VGG(slide_data_path, slide_save_path)
  

    