import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import csv
from collections import Counter
import scipy.spatial.distance
import pickle

def chi_squared_distance(hist1, hist2):
    return np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + 1e-6))

def predict(new_histogram, known_histograms, known_labels, top_n):
    # Calculate the distances
    distances = np.array([chi_squared_distance(new_histogram, h) for h in known_histograms])

    # Find the top_n similar cases   
    similar_indices = np.argsort(distances)
    top_n_labels = np.array(known_labels)[similar_indices][:top_n].tolist()


    # # Find the majority vote
    # majority_vote = Counter(top_n_labels).most_common(1)[0][0]
    # print(majority_vote)
    
    # Find the majority vote
    majority_vote = Counter(top_n_labels).most_common(1)[0][0]
    return majority_vote

def leave_one_out_test(histogram_dict, csv_filepath, top_n):
    # Read the CSV file to get slide names and corresponding labels
    with open(csv_filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        # Or replace with whatever is your filename header 
        #slide_labels = {row['\ufefffile_name']: row["label"] for row in reader}
        slide_labels = {row['file_name']: row["label"] for row in reader}
   

    # Initialize lists to store true and predicted labels
    y_true = []
    y_pred = []
    
    # # Error handling to remove the empty histograms
    # for slide_name, hist in histogram_dict.items():
    #     if hist is None:
    #         del histogram_dict[slide_name]
            

    for slide_name, hist in histogram_dict.items():
        # SLIDE NAME MUST END WITH .svs to do search!!! (or you can change)
        true_label = slide_labels.get(slide_name, None)

        if true_label is not None:
            # Remove the current histogram from the known histograms
            known_histograms = [h for s, h in histogram_dict.items() if s != slide_name] # added the spit for crc
            # Remove the corresponding label from the known labels
            known_labels = [slide_labels.get(s, None) for s in histogram_dict.keys() if s != slide_name] # added the spit for crc

            # Predict the label of the test histogram
            pred_label = predict(hist, known_histograms, known_labels, top_n)

            y_true.append(true_label)
            y_pred.append(pred_label)
        else:
            print(f"Warning: Slide '{slide_name}' not found in the label CSV file.")

    return y_true, y_pred

if __name__ == "__main__":
    # Load the histogram dictionary from the .pkl file
    histogram_path = '/mayo_atlas/home/m296984/visual_dictionary_pipeline/new_pipeline_data/VAE_16dim_32x32/test_1000_slide_dictionary.pkl'
    with open(histogram_path, 'rb') as file:
        histogram_dict = pickle.load(file)

    csv_filepath = '/mayo_atlas/atlas/liver_ash_nash/label/liver_ash_nash_normal_3classes.csv' 
    top_n = 5

    y_true, y_pred = leave_one_out_test(histogram_dict, csv_filepath, top_n)

    print("True Labels:", y_true)
    print("Predicted Labels:", y_pred)

    # Print the confusion matrix and classification report
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))
