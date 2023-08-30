import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import matplotlib as plt
import matplotlib.pyplot as plt
import tqdm as tqdm

def assign_to_clusters(embeddings, visual_dictionary):
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(visual_dictionary)
    _, cluster_indices = knn.kneighbors(embeddings)
    return cluster_indices.flatten()

def save_histogram(input_embeddings_path, visual_dictionary, output_path):
    # Iterate through each slide directory
    for slide_directory in tqdm.tqdm(os.listdir(input_embeddings_path), total= len(os.listdir(input_embeddings_path))):
        slide_path = os.path.join(input_embeddings_path, slide_directory)
        if os.path.isdir(slide_path):
            # Load embeddings for the current slide
            embeddings_list = []
            for embedding_file in os.listdir(slide_path):
                embedding = torch.load(os.path.join(slide_path, embedding_file))
                embedding =  embedding.detach().cpu().numpy()
                embeddings_list.append(embedding)
            embeddings_array = np.array(embeddings_list)

            num_embeddings, embedding_size = embeddings_array.shape[0], np.prod(embeddings_array.shape[1:])
            embeddings_array = embeddings_array.reshape(num_embeddings, embedding_size)
            
            cluster_indices = assign_to_clusters(embeddings_array, visual_dictionary)

            # Generate histogram
            num_clusters = len(visual_dictionary)
            histogram, _ = np.histogram(cluster_indices, bins=np.arange(num_clusters+1))

            # Save the histogram for the current slide
            output_file_path = os.path.join(output_path, f'{slide_directory}_histogram.npy')
            np.save(output_file_path, histogram)
            
if __name__ == "__main__":
    model = 'VAE'
    word_size = 32
    n_words = 500
    keywords = ["test"]
    n_clusters = [128, 256, 512]
    for keyword in keywords: 
        input_embeddings_path = f'/mayo_atlas/home/m296984/visual_dictionary_pipeline/{keyword}/embeddings_{n_words}_{model}_{word_size}x{word_size}'
        for n in n_clusters:
            visual_dictionary = np.load(f'/mayo_atlas/home/m296984/visual_dictionary_pipeline/kmeans_{n_words}_results_{word_size}/visual_dictionary_{n}.npy')
            output_histograms_path = f'/mayo_atlas/home/m296984/visual_dictionary_pipeline/{keyword}/histograms_{n_words}_{word_size}x{word_size}/histogram_{n}'
            # Create the output histogram directory if it doesn't exist
            os.makedirs(output_histograms_path, exist_ok=True)
            print(f'Saving Histogram for the {keyword} set at cluster {n}')
            save_histogram(input_embeddings_path, visual_dictionary, output_histograms_path)
        
        