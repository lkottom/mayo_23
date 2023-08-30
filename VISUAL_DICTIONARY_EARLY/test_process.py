import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import multiprocessing
import csv

def get_label(target_slide_name, csv_filepath):
    with open(csv_filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['\ufefffile_name'] == target_slide_name:
                return row["label"]
        return None
            
def chi_squared_distance(hist1, hist2):
    return np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + 1e-6))

def get_validation_histograms(validation_histo_path):
    
    train_histogram_files = os.listdir(validation_histo_path)

    val_histograms = []
    for file in train_histogram_files:
        if file.endswith(".npy"):
            histogram_path = os.path.join(validation_histo_path, file)
            histogram = np.load(histogram_path)
            val_histograms.append((file, histogram))
            
    return val_histograms
    

def best_match(test_histogram_path, vis_dict_histo, label_filepath):
    test_histogram = np.load(test_histogram_path)
    distances_dict = {}

    for name, histogram in vis_dict_histo:
        distance = chi_squared_distance(test_histogram, histogram)
        distances_dict[name] = distance
        

    # Sort distances_dict by distance in ascending order
    sorted_distances = sorted(distances_dict.items(), key=lambda x: x[1])
    
    label_dictionary = {}
    for k, _ in sorted_distances:
        k = k.replace("_histogram.npy", "")
        slide_name = k + '.svs'
        label = get_label(slide_name, label_filepath)
        label_dictionary[k] = label

    return label_dictionary

def compute_accuracy(slide_label, label_dictionary, top_n):
    top_n_labels = []
    counter = 0
    for name, _ in label_dictionary.items():
        if counter >= top_n:
            break
        else:
            top_n_labels.append(label_dictionary[name])
            counter +=1 
            
    # For top_1, check if slide_name exactly matches the closest prediction
    if top_n == 1:
        if top_n_labels[0] == slide_label:
            return 1
        else:
            return 0
        
    # For top_2 and top_5, use the majority voting mechanism
    count = sum(1 for name in top_n_labels if name == slide_label)
    if count >= top_n // 2:
        return 1
    else:
        return 0
    
def process_file(file, test_histo_dir, val_histograms, label_path):
    histo_path = os.path.join(test_histo_dir, file)
    slide_name = file.replace("_histogram.npy", "")
    slide_name = slide_name + ".svs"
    sorted_labels = best_match(histo_path, val_histograms, label_path)
    test_label = get_label(slide_name, '/mayo_atlas/home/m296984/visual_dictionary_pipeline/liver_ash_nash_365_annotation.csv')

    return compute_accuracy(test_label, sorted_labels, 1), compute_accuracy(test_label, sorted_labels, 2), compute_accuracy(test_label, sorted_labels, 5)

def get_results(test_histo_dir, val_histograms, label_path):
    correct_slide_top1 = 0
    correct_slide_top2 = 0
    correct_slide_top5 = 0
    total = 0

    files = os.listdir(test_histo_dir)

    # Number of processes to use
    num_processes = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(process_file, [(file, test_histo_dir, val_histograms, label_path) for file in files])

    for top1, top2, top5 in results:
        correct_slide_top1 += top1
        correct_slide_top2 += top2
        correct_slide_top5 += top5
        total += 1
    
    accuracy_1 = (correct_slide_top1 / total) * 100
    formatted_accuracy_1 = "{:.2f}".format(accuracy_1)
    accuracy_2 = (correct_slide_top2 / total) * 100
    formatted_accuracy_2 = "{:.2f}".format(accuracy_2)
    accuracy_5 = (correct_slide_top5 / total) * 100
    formatted_accuracy_5 = "{:.2f}".format(accuracy_5)

    print()
    print(f'Top_1 Accuracy: {formatted_accuracy_1}')
    print(f'Top_2 Accuracy: {formatted_accuracy_2}')
    print(f'Top_5 Accuracy: {formatted_accuracy_5}')

    
if __name__ == "__main__":
    n_words = 500
    n_clusters = [128, 256, 512]
    patch_size = 32
    LABEL_PATH = '/mayo_atlas/home/m296984/visual_dictionary_pipeline/liver_ash_nash_365_annotation.csv' 
    for n in n_clusters:
        val_hist_path = f'/mayo_atlas/home/m296984/visual_dictionary_pipeline/train/histograms_{n_words}_{patch_size}x{patch_size}/histogram_{n}'
        test_path = f'/mayo_atlas/home/m296984/visual_dictionary_pipeline/test/histograms_{n_words}_{patch_size}x{patch_size}/histogram_{n}'
        validation_histograms = get_validation_histograms(val_hist_path)
        print(f'---Getting Results for cluster {n}---')
        get_results(test_path, validation_histograms, LABEL_PATH)
        print('--------------------------------------')
   
        
     
    

    
    