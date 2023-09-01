This README file contains a step by step process to create and test the "visual dictionary' style approach 
The best way to reach me for any bugs, code or pipeline questions, or anything else will be through my 
personal email then my school email. All the functions have in-depth docustrings which should hopefully help clear up most questions. 

Personal email: lukekottom@gmail.com
School email: lkottom@caltech.edu

Entire Visual Dictionary Pipeline:

Step 1: Seperate the WSIs into and Train/Test set

    This is done in the copy_slides.py file. We decide the train/test set based on the info in the CSV file, as if we don't have a label or data, we cannot test the slide so it can be used to get the training patches. It is kinda a crude approach to moving files but it has multiple checks to make sure you are copying the correct WSIs to your train/test set. Why I did this was because then you are not working directly with the WSIs data directly in the atlas and protects you from deleting or alternating any of it while testing. 

    This uses the CSV files with the labels to designated which ones should be in the train/test set for each tissue type. The four CSV files used in the testing process is (all should be either included or in the atlas files):

        - Breast_Atlas_Yottixel_2.csv
        - all_CRC_filtered_last_wsi_each_patient.csv
        - liver_ash_nash_normal_3classes.csv
        - skin_well_mod_poor_normal_annotation.csv
    
    The file_name columns would all be in the format "wsi_name.svs" for the following code to work so if you need to ever add ".svs" to any of the files then the code to do that is in the third box in the copy_slides.ipynb if needed. Other then that, this code should correctly divide the slides into a train and test set.


Step 2: Get train patches from the train dataset from the previous step.

    This is done in the quick_patching code. It is pretty straight forward and can be run by pushing the 'play' button in VSCode after the paths and parameters are changed. Just change the number of of words you want from the train set and the patch size, along with where to save them, and you should be good to go. It basically just goes over each slide in the set and grabs the number of "visual words" at random and saves them as .pngs in a directory (which is nice if you want to train multiple differnet training configs later)

    THE MOST IMPORTANT THINGS IS THIS: The number of words_per_slide will be multiplied by 4 because each patch is rotated 360 degrees so make sure you don't get too many patches if you do not want (I found about 1 million works well for training) So the total number of words saved to your directory will be:

     - number_of_WSIs_in_train_dir * words_per_slide * 4

Step 3: Train your model and saved the path, mess with layers/depth, etc....

    This step is done in the model_vae_linear.py, the conv_VAE.py, and the train_and_viz.ipynb. You can basically change anything in the two model .py folders you want. Each have a testing case as the bottom of the file so you can check that your config/changes work by running the 'play' button again. Basically these test just make sure you input and output shapes are matching and you latent shapes is what you want. One is using convolutional layer and the other is just using linear...mess with it and change all you want. 

    After you make the model how you want or add another one, that is when you run the train_and_viz.py folder. This basically loads all the train patched you created from step two and you can mess with the batch size and anything else, though the config right now works well for about 1 million training patches....it takes about an hour to train 10 epchos. 

    It is also a little crude and can use some editing but what it does is after every 500 batches it will save the .png image of 5 inputed "visual words" and how the model reconstructed them, so you can visualize how well it is doing. It alse prints a loss curve after each epcho using MSError (This is the part that could be worked on...I dont think the graph is the best/how it should be set up but it does work)


Step 4: Run the entire visual dictionary pipeline

    This is done in the entire_dictionary_pipeline.py folder. All the functions should work properly, however, you might have to change lines 83-86 based on the "file_name" header in the CSV (sometimes it is'file_name' or '\ufefffile_name' based on the CSV file, don't know why) Other then that, just set the number_words you want and all the filepaths correctly and it should run from test WSIs -> Top1, Top3, Top5 results. None of the functions about the main loop need to be changed and handle all errors that could possibly happen

Additions/Fixes to the code:

    The patching method does need a little work. It misses slide or two based on the fact that it can't find 1024x1024 patches with 70% tissue threshhold but this does not break the code and it just skips it. Adding a GUI would be VERY helpful as the entire pipeline is already established and could be done while running tests on the tissue (I just didn't have enough experience/time to do this). The other major change that is need is building a much more complex model that is deeper and reconstructs the "visual words" better and you can then, after testing, just input that into the pipeline whenever.