import tkinter as tk
import torch
from tkinter import filedialog
from entire_dictionary_pipeline import get_words, get_embeddings, apply_kmeans, get_histograms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_histogram():
    wsi_dir = wsi_dir_entry.get()
    model_path = model_path_entry.get()
    words_per_patch = 50
    patch_size = 32
    cluster_size = 128
    
    slide_dictionary = get_words(wsi_dir, words_per_patch, patch_size)
    slide_dictionary, embeddings_np = get_embeddings(slide_dictionary, model_path, patch_size, device)
    visual_dictionary = apply_kmeans(cluster_size, embeddings_np)
    slide_dictionary = get_histograms(slide_dictionary, visual_dictionary, cluster_size)
    
    # Display the output in the label
    result_label.config(text=str(slide_dictionary))

def browse_wsi_dir():
    wsi_dir = filedialog.askdirectory()
    wsi_dir_entry.delete(0, tk.END)
    wsi_dir_entry.insert(0, wsi_dir)

def browse_model_path():
    model_path = filedialog.askopenfilename()
    model_path_entry.delete(0, tk.END)
    model_path_entry.insert(0, model_path)

# Create the main application window
app = tk.Tk()
app.title("Histogram Generation")

# Create input fields and labels
wsi_dir_label = tk.Label(app, text="WSI Directory:")
wsi_dir_label.pack()
wsi_dir_entry = tk.Entry(app)
wsi_dir_entry.pack()
wsi_dir_button = tk.Button(app, text="Browse", command=browse_wsi_dir)
wsi_dir_button.pack()

model_path_label = tk.Label(app, text="Model Path:")
model_path_label.pack()
model_path_entry = tk.Entry(app)
model_path_entry.pack()
model_path_button = tk.Button(app, text="Browse", command=browse_model_path)
model_path_button.pack()

# Create the button to trigger the process
generate_button = tk.Button(app, text="Generate Histogram", command=generate_histogram)
generate_button.pack()

# Create a label to display the output
result_label = tk.Label(app, text="")
result_label.pack()

# Run the main loop
app.mainloop()