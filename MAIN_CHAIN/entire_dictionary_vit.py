import os
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
from leave_one_out_testing import leave_one_out_test
from quick_patching import process_wsi
from quick_patching_dino import process_wsi_dino
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from model_vae_8_dim import Autoencoder
from torchvision import transforms
import tqdm as tqdm
import time

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def save_results_to_file(file_path, content):
    with open(file_path, 'a') as file:
        file.write(content)
        file.write("\n\n")
        
def load_model(patch_size, model_path):
    model = Autoencoder(patch_size)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model.eval()

def apply_kmeans(num_of_clusters, all_embeddings):
    print(f'There are a total of {len(all_embeddings)} to apply KMeans too')
    print(f"Applying KMeans with {num_of_clusters} clusters....")
    kmeans = KMeans(n_clusters=num_of_clusters, random_state=0, n_init=10).fit(all_embeddings)
    visual_dictionary = kmeans.cluster_centers_
    return visual_dictionary

def assign_to_clusters(embeddings, visual_dictionary):
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(visual_dictionary)
    _, cluster_indices = knn.kneighbors(embeddings)
    return cluster_indices.flatten()

def get_words(wsi_dir, words_per_patch, patch_size):
    slide_dictionary = {}
    for file in tqdm.tqdm(os.listdir(wsi_dir), total=len(os.listdir(wsi_dir)), desc="Getting Words: "):
        if file.endswith('.svs'):
            wsi_path = os.path.join(wsi_dir, file)
            wsi_name = os.path.basename(wsi_path)
            # patch_list = process_wsi(wsi_path, number_of_words_per_patch=words_per_patch, 
            #                          word_size=patch_size, number_of_patches=20, output_patch_size=1024)
            patch_list = process_wsi_dino(wsi_path, number_of_patches = 25)
            if type(patch_list) != bool:
                slide_dictionary[wsi_name] = patch_list
    return slide_dictionary

def get_embeddings(slide_dictionary, model_path, patch_size, device):
    transform = transforms.Compose([
        transforms.Resize(patch_size),
        transforms.ToTensor(),
        # Add any other transformations you need, like normalization
    ])

    model = load_model(patch_size, model_path)
    all_embeddings = []

    for wsi_name, patch_list in tqdm.tqdm(slide_dictionary.items(), total=len(slide_dictionary), desc="Getting Embeddings: "):
        image_embeddings = []
        for image_pil in patch_list:
            # Apply the transformation to the PIL image
            image_tensor = transform(image_pil).unsqueeze(0).to(device)
            _, embedding, _ = model(image_tensor)
            all_embeddings.append(embedding)
            embedding = embedding.detach().cpu().numpy()
            image_embeddings.append(embedding)
        embeddings_array = np.array(image_embeddings)
        slide_dictionary[wsi_name] = embeddings_array
        # print(len(embeddings_array))

    # Concatenate all the individual embeddings into a single tensor
    all_embeddings = torch.cat(all_embeddings, dim=0)
    # Convert the embeddings tensor to a numpy array for clustering
    embeddings_np = all_embeddings.cpu().detach().numpy()
    return slide_dictionary, embeddings_np

def get_embeddings_dino(slide_dictionary, model):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    all_embeddings = []
    
    for wsi_name, patch_list in tqdm.tqdm(slide_dictionary.items(), total=len(slide_dictionary), desc="Getting Embeddings: "):
        image_embeddings = []
        for image_pil in patch_list:
            # Apply the transformation to the PIL image
            image_tensor = preprocess(image_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model(image_tensor)
                all_embeddings.append(embedding)
                embedding = embedding.detach().cpu().numpy()
                image_embeddings.append(embedding)
        embeddings_array = np.array(image_embeddings)
        slide_dictionary[wsi_name] = embeddings_array
        
    # Concatenate all the individual embeddings into a single tensor
    all_embeddings = torch.cat(all_embeddings, dim=0)
    # Convert the embeddings tensor to a numpy array for clustering
    embeddings_np = all_embeddings.cpu().detach().numpy()
    return slide_dictionary, embeddings_np
    

def get_histograms(slide_dictionary, visual_dictionary):
    num_clusters = len(visual_dictionary)

    for wsi_name, embedding_array in tqdm.tqdm(slide_dictionary.items(), total=len(slide_dictionary), desc="Getting Histograms: "):
        num_embeddings, embedding_size = embedding_array.shape[0], np.prod(embedding_array.shape[1:])
        
        # Error handling to check for empty array (In case it did not get any embeddings)
        if embedding_array.size == 0 or np.all(embedding_array == 0):
            slide_dictionary[wsi_name] == None
            continue 
        
        embedding_array = embedding_array.reshape(num_embeddings, embedding_size)
        cluster_indices = assign_to_clusters(embedding_array, visual_dictionary)
        histogram, _ = np.histogram(cluster_indices, bins=np.arange(num_clusters+1))
        slide_dictionary[wsi_name] = histogram

    return slide_dictionary


if __name__ == "__main__":
    start_time = time.time()
    model = 'VAE_8_hdim'
    model = 'dino_2nd'
    model_dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    model_dino.to(device)
    model_dino.eval()
    number_words_per_slide = 25
    # kmean cluster size
    cluster_size = 512
    patch_size = 256
    #model_name = f'{patch_size}x{patch_size}_8_hdim'
    model_name ="vit_16"
    
    
 
    #Your test set of WSI that you want to apply leave one out testing on
    #wsi_dir = '/mayo_atlas/home/m296984/MAIN_CHAIN_LIVER_RESULTS/TEST'
    wsi_dir = '/mayo_atlas/home/m296984/MAIN_CHAIN_LIVER_RESULTS/testing_data'
    
    # Your model path to your pretrained model on the train set. (1 million 16x16 patches is a good amount)
    #model_path = f'/mayo_atlas/home/m296984/RESULTS_40x/Liver/saved_models_{patch_size}/checkpoint_15.pt'
    # CSV file path the must have 'file_name' and 'label' as a header. And the filenames must be "names.svs" (because that is how I searched)
    csv_path = '/mayo_atlas/home/m296984/MAIN_CHAIN_LIVER_RESULTS/liver_ash_nash_normal_3classes.csv' 
    # Where to say your Top 1 Top3 and Top5
    save_data_path = f'/mayo_atlas/home/m296984/RESULTS_DINO/Liver/results_{patch_size}x{patch_size}/{cluster_size}_clusters/{model}_{model_name}_{number_words_per_slide}_patches'
    os.makedirs(save_data_path, exist_ok=True)
    
    words_per_patch = number_words_per_slide/20
    slide_dictionary = get_words(wsi_dir, words_per_patch, patch_size)
    #slide_dictionary, embeddings_np = get_embeddings(slide_dictionary, model_path, patch_size, device)
    slide_dictionary, embeddings_np = get_embeddings_dino(slide_dictionary, model_dino)
    visual_dictionary = apply_kmeans(cluster_size, embeddings_np)
    slide_dictionary = get_histograms(slide_dictionary, visual_dictionary)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Dictionary Creation time: {execution_time:.2f} seconds")
    
    start_time = time.time()
    top_n = [1, 3, 5]
    for n in top_n:
        y_true, y_pred = leave_one_out_test(slide_dictionary, csv_path, n)
        cm = confusion_matrix(y_true, y_pred)
        cr = classification_report(y_true, y_pred, digits=4)

        # Prepare the content for the text file
        content = f"Model Name: {model}_{model_name}\nNumber of Words per Slide: {number_words_per_slide}\nCluster Size: {cluster_size}\nPatch Size: {patch_size}\nTop N: {n}\n\n"
        content += "Confusion Matrix:\n"
        content += str(cm)
        content += "\n\nClassification Report:\n"
        content += cr
        content += "\n\n"
        filename = f"{model}_{patch_size}x{patch_size}_{number_words_per_slide}_top{n}.txt"
        file_path = os.path.join(save_data_path, filename)

        # Save the results to the text file
        save_results_to_file(file_path, content)
        end_time = time.time()
        execution_time = end_time - start_time
    print(f"Testing time: {execution_time:.2f} seconds")
            
    print("Results have been saved!")
    
    