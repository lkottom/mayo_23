import os
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
from leave_one_out_testing import leave_one_out_test
from quick_patching import process_wsi
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from model_vae_linear import Autoencoder
from torchvision import transforms
import tqdm as tqdm
from minisom import MiniSom
import time


device = torch.device("cpu")

def save_results_to_file(file_path, content):
    """
    This function opens the specified file in 'append' mode and writes the
    provided content to it. It also adds two newline characters after the content
    to separate entries.

    Args:
        file_path (str): The path to the file where the content will be saved.
        content (str): The content to be written to the file.

    Returns:
        None


    Example:
        save_results_to_file("output.txt", "This is a sample result.")
    """
    with open(file_path, 'a') as file:
        file.write(content)
        file.write("\n\n")
        
def load_model(patch_size, model_path):
    """
    This function creates an instance of an Autoencoder model with the given patch size,
    loads the model's state dictionary from the provided model path, and moves the
    model to the appropriate device (e.g., GPU) if available. The loaded model is then
    switched to evaluation mode using `model.eval()` and returned.

    Args:
        patch_size (int): The size of the input patches expected by the model.
        model_path (str): The path to the saved model's checkpoint.

    Returns:
        torch.nn.Module: The loaded model in evaluation mode.

    Example:
        loaded_model = load_model(128, "saved_models/autoencoder_checkpoint.pth")
    """
    model = Autoencoder(patch_size)  # Create an instance of the Autoencoder model
    model.load_state_dict(torch.load(model_path))  # Load model's state dictionary
    model.to(device)  # Move model to the appropriate device
    return model.eval()  # Switch model to evaluation mode and return it

def apply_kmeans(num_of_clusters, all_embeddings):
    """
    This function applies KMeans clustering to a set of embeddings using the specified
    number of clusters. It prints information about the process, such as the number of
    embeddings and the number of clusters. The function then fits the KMeans algorithm
    to the embeddings, calculates cluster centers, and returns them.

    Args:
        num_of_clusters (int): The desired number of clusters.
        all_embeddings (numpy.ndarray): An array of embeddings.

    Returns:
        numpy.ndarray: Cluster centers obtained from KMeans.

    Example:
        embeddings = load_embeddings("embedding_data.npy")
        clusters = apply_kmeans(10, embeddings)
    """

    print(f'There are a total of {len(all_embeddings)} embeddings to apply KMeans too')
    print(f"Applying KMeans with {num_of_clusters} clusters....")
    kmeans = KMeans(n_clusters=num_of_clusters, random_state=0, n_init=10).fit(all_embeddings)
    visual_dictionary = kmeans.cluster_centers_
    return visual_dictionary

def assign_to_clusters(embeddings, visual_dictionary):
    """
    This function utilizes the Nearest Neighbors algorithm to assign embeddings to
    clusters based on their nearest points in the provided visual dictionary. It fits
    a Nearest Neighbors model to the visual dictionary and then finds the nearest
    cluster center for each embedding. The resulting cluster indices are returned.

    Args:
        embeddings (numpy.ndarray): The embeddings to be assigned.
        visual_dictionary (numpy.ndarray): The cluster centers.

    Returns:
        numpy.ndarray: Cluster indices assigned to the embeddings (flattened).

    Example:
        embeddings = load_embeddings("embedding_data.npy")
        clusters = load_clusters("cluster_centers.npy")
        assigned_indices = assign_to_clusters(embeddings, clusters)
    
    """
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(visual_dictionary)
    _, cluster_indices = knn.kneighbors(embeddings)
    return cluster_indices.flatten()

def get_words(wsi_dir, words_per_patch, patch_size):
    """
    This function processes WSI files in the specified directory, extracting word patches
    from them. It uses the `process_wsi` function to extract patches from each WSI and
    compiles them into a dictionary. The keys of the dictionary are WSI names, and the
    values are lists of word patches extracted from the respective WSIs.

    Args:
        wsi_dir (str): Path to the directory containing WSI files.
        words_per_patch (int): Number of words to extract from each patch.
        patch_size (int): Size of the patches to be extracted.

    Returns:
        dict: A dictionary with WSI names as keys and lists of word patches as values.

    Example:
        wsi_directory = "WSI_data"
        words_per_patch = 5
        patch_size = 224
        word_patches_dict = get_words(wsi_directory, words_per_patch, patch_size)
    """
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
    """
    This function processes patches in the provided slide dictionary using a pre-trained model.
    It applies the specified transformations to the patches, extracts embeddings using the model,
    and stores the embeddings in the slide dictionary. It also concatenates all the individual
    embeddings into a single array for subsequent clustering.

    Args:
        slide_dictionary (dict): Dictionary with slide names as keys and patch lists as values.
        model_path (str): Path to the pre-trained model's checkpoint.
        patch_size (int): Size of patches to be processed by the model.
        device: Device on which the model should run (e.g., "cuda" or "cpu").

    Returns:
        dict: Updated slide dictionary with embeddings arrays.
        numpy.ndarray: Array containing all extracted embeddings.

    Example:
        wsi_embeddings_dict, all_embeddings = get_embeddings(slide_dict, "autoencoder.pth", 128, "cuda")
    """
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

def get_histograms(slide_dictionary, visual_dictionary):
    """
    This function calculates histograms of cluster assignments for each slide based on their embeddings.
    It uses the provided visual dictionary to map embeddings to cluster indices and computes histograms
    of these assignments. The updated histograms are stored in the slide dictionary.

    Args:
        slide_dictionary (dict): Dictionary with slide names as keys and embeddings histograms as values.
        visual_dictionary (numpy.ndarray): Visual dictionary obtained from KMeans clustering.

    Returns:
        dict: Updated slide dictionary with computed histograms.

    Example:
        histograms_dict = get_histograms(slide_dict, visual_dictionary)
    """
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
    model = 'VAE_32_hdim'
    number_words_per_slide = 4000
    # kmean cluster size
    cluster_size = 2048
    patch_size = 16
    model_name = f'{patch_size}x{patch_size}_8_hdim'
    # som_size = (250, 250)  # Size of the SOM grid, you can adjust it as needed
    
 
    # Your test set of WSI that you want to apply leave one out testing on
    #wsi_dir = '/mayo_atlas/home/m296984/MAIN_CHAIN_BREAST_RESULTS/test_dataset'
    wsi_dir = '/mayo_atlas/home/m296984/MAIN_CHAIN_BREAST_RESULTS/testing_data'
    
    # Your model path to your pretrained model on the train set. (1 million 16x16 patches is a good amount)
    model_path = f'/mayo_atlas/home/m296984/VAE_CHAIN/BREAST/saved_models_16_abs/autoencoder_epoch_9.pth'
    # CSV file path the must have 'file_name' and 'label' as a header. And the filenames must be "names.svs" (because that is how I searched)
    #csv_path = '/mayo_atlas/home/m296984/MAIN_CHAIN_LIVER_RESULTS/liver_ash_nash_normal_3classes.csv' 
    csv_path = '/mayo_atlas/home/m296984/MAIN_CHAIN_BREAST_RESULTS/Breast_Atlas_Yottixel_2.csv'
    #csv_path = '/mayo_atlas/home/m296984/MAIN_CHAIN_SKIN_RESULTS/skin_well_mod_poor_normal_annotation.csv'
    #csv_path = '/mayo_atlas/home/m296984/MAIN_CHAIN_CRC_RESULTS/all_CRC_filtered_last_wsi_each_patient.csv'
    
    # Where to save your Top1, Top3, and Top5
    save_data_path = f'/mayo_atlas/home/m296984/VAE_CHAIN/BREAST/results_abstract_2969num2/{cluster_size}_clusters/{model}_{model_name}_{number_words_per_slide}'
    os.makedirs(save_data_path, exist_ok=True)
    
    # Running the entire dictionary process
    words_per_patch = number_words_per_slide/20
    slide_dictionary = get_words(wsi_dir, words_per_patch, patch_size)
    slide_dictionary, embeddings_np = get_embeddings(slide_dictionary, model_path, patch_size, device)
    visual_dictionary = apply_kmeans(cluster_size, embeddings_np)
    slide_dictionary = get_histograms(slide_dictionary, visual_dictionary)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Dictionary Creation time: {execution_time:.2f} seconds")
    
    # # Use MiniSom clustering
    # visual_dictionary = apply_minisom(cluster_size, embeddings_np, som_size)
    # slide_dictionary = get_histograms_minisom(slide_dictionary, visual_dictionary)
    
    # Getting the leave-one-out testing results 
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
    
    