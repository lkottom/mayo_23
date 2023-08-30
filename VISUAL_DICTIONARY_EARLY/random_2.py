import os
from tqdm import tqdm

directory_path = '/mayo_atlas/home/m296984/visual_dictionary_pipeline/training_32x32_patches'

def delete_directory(directory):
    try:
        os.rmdir(directory)
        return True
    except OSError:
        return False

if __name__ == "__main__":
    target_directory = os.path.join(directory_path, 'vae_logs')

    if os.path.exists(target_directory):
        print("Directory 'vae_logs' found. Deleting...")
        for _ in tqdm(range(10)):
            if delete_directory(target_directory):
                print("Directory 'vae_logs' deleted successfully.")
                break
    else:
        print("Directory 'vae_logs' not found in the specified path.")
