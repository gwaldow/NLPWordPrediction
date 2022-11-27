The word prediction model in this program possesses the capability of predicting single next words given user input string of arbitrary length. This process can be stringed together to generate sentences from user input as well. Our model uses 4-grams for predictions and implements smoothing techniques and stop words. 

Before the program can operate correctly, it requires training data. The quantity of data needed to train this type of model is large, and as such the training data file cannot be stored on github. Use this ( https://www.kaggle.com/crmercado/tweets-blogs-news-swiftkey-dataset-4million ) link to download the kaggle dataset "en_US.blogs.txt" (other datasets may be used but they may not yeild the same accuracy). Then, rename this file to "train.txt" and put it into the program's data folder.

To run the demonstration and test of the program, use your prefered command terminal and navigate into the folder "AI_FINAL_PROJECT_NEXT_WORD_PREDICTOR". Then, enter "python main.py" into the command line. This will begin the program.

What the Program Does:

1. The program will generate files containing n-gram data using the training data found in "data/train.txt". This training data is has a large filesize, and cannot be stored on github. The data we used can be found here ( https://www.kaggle.com/crmercado/tweets-blogs-news-swiftkey-dataset-4million ).

If n-gram data has already been generated from the training data, the program will simply load in the .pkl files found in the "saved" folder.

2. The program will then ask the user for an input. The user may enter any word or phrase and the program will return it's prediction for the next word. This is similar in function to predictive text.

3. The programm will then ask the user for an input yet again. The type of user input remains the same, however, the program will now generate it's prediction for the next 40 words following the user's word or phrase.

4. Lastly, the program will run an accuracy test using the data in "data/test.txt". By default, this file will be uploaded to the github with 1000 phrases to test our model with. This test will evaluate the model's accuracy using the first 100 phrases and may take a few minutes. It then returns the accuracy of our model in predicting the exact final word of each phrase.

Note: When testing accuracy with n=1000 phrases, our model has shown 12.6% accuracy in predicting the exact next word using maximum ngram size of 4.

Descriptions of the methods of the program are provided in the source code.

Packages Used:

pickle - file reading and writing package in the Python standard library which we use to read and write n-gram data. This is the only function of the package in our code.

collections - package from the Python standard library which implements a data structure similar to a dictionary. This data structure is used in our n_most_common_n_grams() function for visualization of our data. This function is not utilized as a part of our word prediction model. It was only used to visualize the data during the development process.
