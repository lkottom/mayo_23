import os
import numpy as np
import torch
import pickle
from sklearn.metrics import confusion_matrix, classification_report
from leave_one_out_testing import leave_one_out_test
from quick_patching import process_wsi
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from model_vae_16_dim import Autoencoder
from torchvision import transforms
import tqdm as tqdm
from minisom import MiniSom

device = ("cuda:2" if torch.cuda.is_available() else "cpu")

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
            patch_list = process_wsi(wsi_path, number_of_words_per_patch=words_per_patch, 
                                     word_size=patch_size, number_of_patches=20, output_patch_size=1024)
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

def get_histograms(slide_dictionary, visual_dictionary, cluster_size):
    num_clusters = len(visual_dictionary)

    for wsi_name, embedding_array in tqdm.tqdm(slide_dictionary.items(), total=len(slide_dictionary), desc="Getting Histograms: "):
        num_embeddings, embedding_size = embedding_array.shape[0], np.prod(embedding_array.shape[1:])
        embedding_array = embedding_array.reshape(num_embeddings, embedding_size)
        cluster_indices = assign_to_clusters(embedding_array, visual_dictionary)
        histogram, _ = np.histogram(cluster_indices, bins=np.arange(num_clusters+1))
        slide_dictionary[wsi_name] = histogram

    return slide_dictionary

def apply_minisom(num_of_clusters, all_embeddings, som_size, init_weights=None):
    print(f'Training MiniSom with {num_of_clusters} clusters....')
    som = MiniSom(som_size[0], som_size[1], all_embeddings.shape[1], sigma=0.5, learning_rate=0.5, random_seed=42)

    if init_weights is not None:
        som._weights = init_weights

    som.train_batch(all_embeddings, 10000, verbose=True)  # You can adjust the number of epochs (10000 here) as needed
    visual_dictionary = som.get_weights().reshape(-1, all_embeddings.shape[1])
    return visual_dictionary

def assign_to_clusters_minisom(embeddings, visual_dictionary):
    cluster_indices = np.array([np.argmin(np.linalg.norm(embeddings - v, axis=1)) for v in visual_dictionary])
    return cluster_indices

def get_histograms_minisom(slide_dictionary, visual_dictionary):
    num_clusters = len(visual_dictionary)

    for wsi_name, embedding_array in tqdm.tqdm(slide_dictionary.items(), total=len(slide_dictionary), desc="Getting Histograms (MiniSom): "):
        num_embeddings, embedding_size = embedding_array.shape[0], np.prod(embedding_array.shape[1:])
        embedding_array = embedding_array.reshape(num_embeddings, embedding_size)
        cluster_indices = assign_to_clusters_minisom(embedding_array, visual_dictionary)
        histogram, _ = np.histogram(cluster_indices, bins=np.arange(num_clusters + 1))
        slide_dictionary[wsi_name] = histogram

    return slide_dictionary


if __name__ == "__main__":
    model = 'TESTTT'
    number_words_per_slide = 1000
    cluster_size = 500
    patch_size = 16
    model_name = f'{patch_size}x{patch_size}_16_hdim'
    som_size = (50, 50)  # Size of the SOM grid, you can adjust it as needed
    
 
    top_n = [1, 2, 5]
    #wsi_dir = '/mayo_atlas/home/m296984/trial_data'
    wsi_dir = f'/mayo_atlas/home/m296984/MAIN_CHAIN/test_liver_data'
    model_path = f'/mayo_atlas/home/m296984/MAIN_CHAIN/saved_models/{model_name}/checkpoint_15.pt'
    csv_path = '/mayo_atlas/home/m296984/visual_dictionary_pipeline/liver_ash_nash_365_annotation.csv' 
    # edited this below line when done running
    save_data_path = f'/mayo_atlas/home/m296984/MAIN_CHAIN/results/{cluster_size}_clusters/{model}_{model_name}_{number_words_per_slide}'
    os.makedirs(save_data_path, exist_ok=True)
    
    words_per_patch = number_words_per_slide/20
    slide_dictionary = get_words(wsi_dir, words_per_patch, patch_size)
    slide_dictionary, embeddings_np = get_embeddings(slide_dictionary, model_path, patch_size, device)
    visual_dictionary_kmeans = apply_kmeans(cluster_size, embeddings_np)

    # Use MiniSom clustering with the visual dictionary from KMeans as initialization
    visual_dictionary_minisom = apply_minisom(cluster_size, embeddings_np, som_size, init_weights=visual_dictionary_kmeans)

    # Get histograms using the MiniSom clustering
    slide_dictionary = get_histograms_minisom(slide_dictionary, visual_dictionary_minisom)
    
    for n in top_n:
        y_true, y_pred = leave_one_out_test(slide_dictionary, csv_path, n)
        cm = confusion_matrix(y_true, y_pred)
        cr = classification_report(y_true, y_pred)

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
        
    print("Results have been saved!")
    
    
                
            
                
   
