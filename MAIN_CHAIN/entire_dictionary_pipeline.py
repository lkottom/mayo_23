# import os
# import numpy as np
# import torch
# from sklearn.metrics import confusion_matrix, classification_report
# from leave_one_out_testing import leave_one_out_test
# from quick_patching import process_wsi
# from sklearn.neighbors import NearestNeighbors
# from sklearn.cluster import KMeans
# from conv_VAE import Autoencoder
# from torchvision import transforms
# import tqdm as tqdm
# import time

# #device = ("cuda:1" if torch.cuda.is_available() else "cpu")    
# device = torch.device("cpu")

# def save_results_to_file(file_path, content):
#     """
#     Save content to a file at the specified path.

#     Parameters:
#     file_path (str): Path to the file where content will be saved.
#     content (str): Content to be written to the file.

#     Returns:
#     None
#     """
#     with open(file_path, 'a') as file:
#         file.write(content)
#         file.write("\n\n")
        
# def load_model(model_path, device):
#     """
#     Load a pre-trained model from a specified path.

#     Parameters:
#     model_path (str): Path to the model's saved state dictionary.
#     device (str): Device to load the model onto.

#     Returns:
#     torch.nn.Module: Loaded model.
#     """
#     model = Autoencoder()  # Create an instance of the model (you may need to replace 'Autoencoder' with the actual model class)
#     state_dict = torch.load(model_path, map_location=device)
    
#     # Handle 'module' prefix if the model was trained using nn.DataParallel
#     if list(state_dict.keys())[0].startswith('module.'):
#         state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
#     model.load_state_dict(state_dict)
#     model.to(device).eval()  # Move the model to the specified device and set to evaluation mode
#     return model

# def apply_kmeans(num_of_clusters, all_embeddings):
#     """
#     Apply KMeans clustering to the provided embeddings.

#     Parameters:
#     num_of_clusters (int): Number of clusters to create.
#     all_embeddings (numpy.ndarray): Array of embeddings for clustering.

#     Returns:
#     numpy.ndarray: The visual dictionary obtained from KMeans clustering.
#     """
#     print(f'There are a total of {len(all_embeddings)} to apply KMeans to')
#     print(f"Applying KMeans with {num_of_clusters} clusters....")
#     kmeans = KMeans(n_clusters=num_of_clusters, random_state=0, n_init=10).fit(all_embeddings)
#     visual_dictionary = kmeans.cluster_centers_
#     return visual_dictionary

# def assign_to_clusters(embeddings, visual_dictionary):
#     """
#     Assign embeddings to clusters based on the nearest neighbors in the visual dictionary.

#     Parameters:
#     embeddings (numpy.ndarray): Array of embeddings to be assigned to clusters.
#     visual_dictionary (numpy.ndarray): Visual dictionary obtained from clustering.

#     Returns:
#     numpy.ndarray: Array of cluster indices corresponding to each embedding.
#     """
#     knn = NearestNeighbors(n_neighbors=1)
#     knn.fit(visual_dictionary)
#     _, cluster_indices = knn.kneighbors(embeddings)
#     return cluster_indices.flatten()

# def get_words(wsi_dir, words_per_patch, patch_size):
#     """
#     Process whole-slide images in a directory to extract words.

#     Parameters:
#     wsi_dir (str): Path to the directory containing whole-slide images.
#     words_per_patch (int): Number of words to extract per patch.
#     patch_size (int): Size of the patches to be extracted.

#     Returns:
#     dict: A dictionary mapping whole-slide image names to lists of extracted patches.
#     """

#     slide_dictionary = {}  # Initialize a dictionary to store extracted patches for each slide

#     # Iterate through each file in the specified directory
#     for file in tqdm.tqdm(os.listdir(wsi_dir), total=len(os.listdir(wsi_dir)), desc="Getting Words: "):
#         if file.endswith('.svs'):
#             wsi_path = os.path.join(wsi_dir, file)
#             wsi_name = os.path.basename(wsi_path)
            
#             # Process the whole-slide image to extract patches
#             patch_list = process_wsi(wsi_path, number_of_words_per_patch=words_per_patch, 
#                                      word_size=patch_size, number_of_patches=20, output_patch_size=1024)
            
#             # Check if valid patches were extracted
#             if type(patch_list) != bool:
#                 slide_dictionary[wsi_name] = patch_list
                
#     return slide_dictionary

# def get_embeddings(slide_dictionary, model_path, patch_size, device):
#     """
#     Obtain embeddings for patches extracted from whole-slide images.

#     Parameters:
#     slide_dictionary (dict): Dictionary mapping whole-slide image names to lists of extracted patches.
#     model_path (str): Path to the pre-trained model for obtaining embeddings.
#     patch_size (int): Size of the patches used for obtaining embeddings.
#     device (str): Device to perform computations on.

#     Returns:
#     dict: Updated slide_dictionary with embeddings for each patch.
#     numpy.ndarray: Numpy array containing all embeddings.
#     """

#     transform = transforms.Compose([
#         transforms.ToTensor(), 
#         transforms.Normalize((0.5,), (0.5,))
#     ])

#     model = load_model(model_path, device)
#     all_embeddings = []

#     # Iterate through each whole-slide image and its corresponding patch list
#     for wsi_name, patch_list in tqdm.tqdm(slide_dictionary.items(), total=len(slide_dictionary), desc="Getting Embeddings: "):
#         image_embeddings = []

#         # Iterate through each patch in the patch list
#         for image_pil in patch_list:
#             # Apply the transformation to the PIL image
#             image_tensor = transform(image_pil).unsqueeze(0).to(device)
            
#             # Obtain the encoded output from the model's encoder
#             encoded_output = model.encoder(image_tensor)
            
#             # Obtain the embedding from the model's latent layer
#             embedding = model.latent_layer(encoded_output)
            
#             # Append the embedding to the list of all_embeddings
#             all_embeddings.append(embedding)
            
#             # Convert the embedding to a numpy array and store it in image_embeddings
#             embedding = embedding.detach().cpu().numpy()
#             image_embeddings.append(embedding)
        
#         # Convert the list of image_embeddings to a numpy array
#         embeddings_array = np.array(image_embeddings)
        
#         # Update the slide_dictionary with the embeddings array
#         slide_dictionary[wsi_name] = embeddings_array

#     # Concatenate all the individual embeddings into a single tensor
#     all_embeddings = torch.cat(all_embeddings, dim=0)
    
#     # Convert the embeddings tensor to a numpy array for clustering
#     embeddings_np = all_embeddings.cpu().detach().numpy()
    
#     return slide_dictionary, embeddings_np


# def get_histograms(slide_dictionary, visual_dictionary):
#     """
#     Calculate histograms of assigned clusters for each whole-slide image.

#     Parameters:
#     slide_dictionary (dict): Dictionary mapping whole-slide image names to embeddings.
#     visual_dictionary (numpy.ndarray): Visual dictionary obtained from clustering.

#     Returns:
#     dict: Updated slide_dictionary with histograms of assigned clusters.
#     """

#     num_clusters = len(visual_dictionary)

#     # Iterate through each whole-slide image and its corresponding embedding array
#     for wsi_name, embedding_array in tqdm.tqdm(slide_dictionary.items(), total=len(slide_dictionary), desc="Getting Histograms: "):
#         num_embeddings, embedding_size = embedding_array.shape[0], np.prod(embedding_array.shape[1:])
        
#         # Reshape the embedding array to have a 2D shape
#         embedding_array = embedding_array.reshape(num_embeddings, embedding_size)
        
#         # Assign embeddings to clusters and calculate histogram
#         cluster_indices = assign_to_clusters(embedding_array, visual_dictionary)
#         histogram, _ = np.histogram(cluster_indices, bins=np.arange(num_clusters+1))
        
#         # Update the slide_dictionary with the histogram
#         slide_dictionary[wsi_name] = histogram

#     return slide_dictionary



# if __name__ == "__main__":
#     start_time = time.time()
#     model = 'VAE_8_hdim_masked'
#     number_words_per_slide = 2000
#     # kmean cluster size
#     cluster_size = 1024
#     patch_size = 32
#     model_name = f'{patch_size}x{patch_size}'
    
 
#     # Your test set of WSI that you want to apply leave one out testing on
#     #wsi_dir = '/mayo_atlas/home/m296984/MAIN_CHAIN_CRC_RESULTS/TEST'
#     wsi_dir = '/mayo_atlas/home/m296984/MAIN_CHAIN_CRC_RESULTS/testing_data'
    
#     # Your model path to your pretrained model on the train set. (1 million 16x16 patches is a good amount)
#     model_path = f'/mayo_atlas/home/m296984/VAE_CHAIN/CRC/32x32/saved_models_8hdim_edited_masked/autoencoder_epoch_9.pth'
#     # CSV file path the must have 'file_name' and 'label' as a header. And the filenames must be "names.svs" (because that is how I searched)
#     csv_path = '/mayo_atlas/home/m296984/MAIN_CHAIN_CRC_RESULTS/CRC_ash_nash_normal_3classes.csv' 
#     # Where to say your Top 1 Top3 and Top5
#     save_data_path = f'/mayo_atlas/home/m296984/VAE_CHAIN/CRC/32x32/saved_models_8hdim_edited_masked/results/{cluster_size}_clusters/{model}_{model_name}_{number_words_per_slide}'
#     os.makedirs(save_data_path, exist_ok=True)
    
#     words_per_patch = number_words_per_slide/20
#     slide_dictionary = get_words(wsi_dir, words_per_patch, patch_size)
#     slide_dictionary, embeddings_np = get_embeddings(slide_dictionary, model_path, patch_size, device)
#     visual_dictionary = apply_kmeans(cluster_size, embeddings_np)
#     slide_dictionary = get_histograms(slide_dictionary, visual_dictionary)
    
#     end_time = time.time()
#     execution_time = end_time - start_time
#     print(f"Dictionary Creation time: {execution_time:.2f} seconds")
    
    
#     start_time = time.time()
#     top_n = [1, 3, 5]
#     for n in top_n:
#         y_true, y_pred = leave_one_out_test(slide_dictionary, csv_path, n)
#         cm = confusion_matrix(y_true, y_pred)
#         cr = classification_report(y_true, y_pred, digits=4)

#         # Prepare the content for the text file
#         content = f"Model Name: {model}_{model_name}\nNumber of Words per Slide: {number_words_per_slide}\nCluster Size: {cluster_size}\nPatch Size: {patch_size}\nTop N: {n}\n\n"
#         content += "Confusion Matrix:\n"
#         content += str(cm)
#         content += "\n\nClassification Report:\n"
#         content += cr
#         content += "\n\n"
#         filename = f"{model}_{patch_size}x{patch_size}_{number_words_per_slide}_top{n}.txt"
#         file_path = os.path.join(save_data_path, filename)

#         # Save the results to the text file
#         save_results_to_file(file_path, content)
#         end_time = time.time()
#         execution_time = end_time - start_time
#     print(f"Testing time: {execution_time:.2f} seconds")
            
#     print("Results have been saved!")

import os
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
from leave_one_out_testing import leave_one_out_test
from quick_patching import process_wsi
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from model_vae_8_dim import Autoencoder
from torchvision import transforms
import tqdm as tqdm
from minisom import MiniSom
import time


device = torch.device("cpu")

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
            embedding, _ = model(image_tensor)
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

# def apply_minisom(num_of_clusters, all_embeddings, som_size):
#     print(f'Training MiniSom with {num_of_clusters} clusters....')
#     som = MiniSom(som_size[0], som_size[1], all_embeddings.shape[1], sigma=0.5, learning_rate=0.5, random_seed=42)
#     som.train_batch(all_embeddings, 10000, verbose=True)  # You can adjust the number of epochs (1000 here) as needed
#     visual_dictionary = som.get_weights().reshape(-1, all_embeddings.shape[1])
#     return visual_dictionary

# def assign_to_clusters_minisom(embeddings, visual_dictionary):
#     cluster_indices = np.array([np.argmin(np.linalg.norm(embeddings - v, axis=1)) for v in visual_dictionary])
#     return cluster_indices

# def get_histograms_minisom(slide_dictionary, visual_dictionary):
#     num_clusters = len(visual_dictionary)

#     for wsi_name, embedding_array in tqdm.tqdm(slide_dictionary.items(), total=len(slide_dictionary), desc="Getting Histograms (MiniSom): "):
#         num_embeddings, embedding_size = embedding_array.shape[0], np.prod(embedding_array.shape[1:])
#         embedding_array = embedding_array.reshape(num_embeddings, embedding_size)
#         cluster_indices = assign_to_clusters_minisom(embedding_array, visual_dictionary)
#         histogram, _ = np.histogram(cluster_indices, bins=np.arange(num_clusters + 1))
#         slide_dictionary[wsi_name] = histogram

#     return slide_dictionary


if __name__ == "__main__":
    start_time = time.time()
    model = 'VAE_8_hdim'
    number_words_per_slide = 2000
    # kmean cluster size
    cluster_size = 1024
    patch_size = 16
    model_name = f'{patch_size}x{patch_size}_8_hdim'
    # som_size = (250, 250)  # Size of the SOM grid, you can adjust it as needed
    
 
    # Your test set of WSI that you want to apply leave one out testing on
    #wsi_dir = '/mayo_atlas/home/m296984/MAIN_CHAIN_CRC_RESULTS/test_dataset'
    wsi_dir = '/mayo_atlas/home/m296984/MAIN_CHAIN_CRC_RESULTS/testing_data'
    
    # Your model path to your pretrained model on the train set. (1 million 16x16 patches is a good amount)
    model_path = f'/mayo_atlas/home/m296984/VAE_CHAIN/CRC/saved_models_8_abs/autoencoder_epoch_1.pth'
    # CSV file path the must have 'file_name' and 'label' as a header. And the filenames must be "names.svs" (because that is how I searched)
    # csv_path = '/mayo_atlas/home/m296984/MAIN_CHAIN_LIVER_RESULTS/liver_well_mod_poor_normal_annotation.csv' 
    # csv_path = '/mayo_atlas/home/m296984/MAIN_CHAIN_BREAST_RESULTS/Breast_Atlas_Yottixel_2.csv'
    # csv_path = '/mayo_atlas/home/m296984/MAIN_CHAIN_SKIN_RESULTS/skin_well_mod_poor_normal_annotation.csv'
    csv_path = '/mayo_atlas/home/m296984/MAIN_CHAIN_CRC_RESULTS/all_CRC_filtered_last_wsi_each_patient.csv'
    # Where to say your Top 1 Top3 and Top5
    save_data_path = f'/mayo_atlas/home/m296984/VAE_CHAIN/CRC/results_abstract/{cluster_size}_clusters/{model}_{model_name}_{number_words_per_slide}'
    os.makedirs(save_data_path, exist_ok=True)
    
    words_per_patch = number_words_per_slide/20
    slide_dictionary = get_words(wsi_dir, words_per_patch, patch_size)
    slide_dictionary, embeddings_np = get_embeddings(slide_dictionary, model_path, patch_size, device)
    visual_dictionary = apply_kmeans(cluster_size, embeddings_np)
    slide_dictionary = get_histograms(slide_dictionary, visual_dictionary, cluster_size)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Dictionary Creation time: {execution_time:.2f} seconds")
    
    # # Use MiniSom clustering
    # visual_dictionary = apply_minisom(cluster_size, embeddings_np, som_size)
    # slide_dictionary = get_histograms_minisom(slide_dictionary, visual_dictionary)
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
    
    