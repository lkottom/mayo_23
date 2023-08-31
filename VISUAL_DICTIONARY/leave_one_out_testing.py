import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import csv
from collections import Counter
import scipy.spatial.distance
import pickle

def chi_squared_distance(hist1, hist2):
    """
    Applies the Chi-squared distance computation to two histograms. 

    Args:
        hist1 (numpy.ndarray): A numpy array representation of histogram 1
        hist2 (numpy.ndarray): A numpy array representation of histogram 2

    Returns:
        float: The chi-squared distance/similarity of the two histograms
    """
    return np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + 1e-6))

def predict(new_histogram, known_histograms, known_labels, top_n):
    """
    This function predicts the label of a new histogram by comparing it to histograms of
    known cases using the chi-squared distance metric. It calculates distances to all
    known histograms, selects the top_n similar cases, and performs a majority vote
    among their labels to determine the predicted label for the new case.

    Args:
        new_histogram (numpy.ndarray): Histogram of the new case.
        known_histograms (list[numpy.ndarray]): Histograms of known cases.
        known_labels (list): Labels of known cases.
        top_n (int): Number of top similar cases to consider.

    Returns:
        Any: Predicted label for the new case.

    Example:
        new_hist = np.array([10, 5, 3, 8, 7])
        known_hists = [np.array([8, 4, 2, 7, 6]), np.array([9, 5, 2, 9, 6])]
        known_lbls = ['Healthy', 'Diseased']
        top_n = 2
        prediction = predict(new_hist, known_hists, known_lbls, top_n)
    """
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
    """
    
    This function applies leave-one-out testing on the entire "visual dictionary". 
    It takes in a CSV file that has the file_name (ex. 35233.svs) and its corresponding 
    label and creates a dictionary of {slide_names: true labels}. It then uses the top_n and the 
    predict function to run through and computed the predicted label based of the slides histogram. 

    Args:
        histogram_dict (dict): Dictionary with slide names as keys and the numpy.ndarray histograms and values
        csv_filepath (str): A file path to the CSV file with the corresponding true labels for each slide
        top_n (int): The top-n vote to apply to the majority vote function 

    Returns:
        _type_: _description_
    """
    # Read the CSV file to get slide names and corresponding labels
    with open(csv_filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        # Or replace with whatever is your filename header 
        slide_labels = {row['\ufefffile_name']: row["label"] for row in reader}
        #slide_labels = {row['file_name']: row["label"] for row in reader}
   

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
    # This code below is for testing purpose only 
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
