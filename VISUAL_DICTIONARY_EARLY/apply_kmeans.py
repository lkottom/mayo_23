import os
import numpy as np
from sklearn.cluster import KMeans
import torch
import tqdm as tqdm



def get_embeddings(dir_path):
    all_embeddings = []
    for sub_dirs in tqdm.tqdm(os.listdir(dir_path), total=len(os.listdir(dir_path))):
        sub_dir_path = os.path.join(dir_path, sub_dirs)
        for subdir in os.listdir(sub_dir_path):
            embedding_path = os.path.join(sub_dir_path, subdir)
            embedding = torch.load(embedding_path, map_location=torch.device('cpu'))
            all_embeddings.append(embedding)
    return all_embeddings


if __name__ == "__main__":
    model = 'VAE_32x32_16_dim'
    word_size = 32
    number_of_words = 500
    n_clusters = [128, 256, 512, 1024, 2048]
    EMBEDDING_PATH = f'/mayo_atlas/home/m296984/visual_dictionary_pipeline/train/embeddings_{number_of_words}_{model}'
    KMEANS_SAVE_DIR = f'/mayo_atlas/home/m296984/visual_dictionary_pipeline/kmeans_{number_of_words}_results_{word_size}'
    os.makedirs(KMEANS_SAVE_DIR, exist_ok=True)
    all_embeddings = get_embeddings(EMBEDDING_PATH)
    print(f'There are a total of {len(all_embeddings)} to apply K_means too')
    # Concatenate all the individual embeddings into a single tensor
    all_embeddings = torch.cat(all_embeddings, dim=0)
    # Convert the embeddings tensor to a numpy array for clustering
    embeddings_np = all_embeddings.cpu().detach().numpy()
    
    for n in tqdm.tqdm(n_clusters, total=len(n_clusters), desc="Getting Kmean Cluster Values: "):
        kmeans = KMeans(n_clusters=n, random_state=0, n_init=10).fit(embeddings_np)
        visual_dictionary = kmeans.cluster_centers_
        save_path = f'{KMEANS_SAVE_DIR}/visual_dictionary_{n}.npy'
        np.save(save_path, visual_dictionary)
    