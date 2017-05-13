<p align = 'center'>In-depth Question Classification using Convolutional Neural Networks <br>
Prudhvi Raj Dachapally and Srikanth Ramanam <br>
Code Execution Instructions <br></p>

1. Data Separation

Before starting the training procedure, to make things easier, we first separate the data based on their classes. That is, the training folder contains 6 different text files for each broader class and each file has their own questions. For example, abbr.txt has questions about abbreviations, etc. 

The data in the training set and test set are in the form of

PRIMARY:secondary --------question---------------
Example:
HUM:ind Who was The Pride of the Yankees ?

The program that does the separation is called main_categ_sep.py. Make sure that you create two directors train_5500 and test_500 before executing this part of the code. This program takes two arguments as input.

	Argument 1: data file (train_5500.txt or test_data.txt)
	Argument 2: Directory name (train_5500 if train, else test_500)

	Example : python main_categ_sep.py train_5500.txt train_5000

This should fill the train_5500  with text files for individual classes.
For separating the classes of the second level categories, the program sub_cate_sep_test.py is used.
Before running that, please create two folders name sub_categories and sub_categories_test.

This does almost the same as the above. But instead of grouping on the primary categories, it groups questions of secondary classes, based on the given primary category.

This program again takes three arguments. 
	
	Argument 1: data file (train_5500.txt or test_data.txt)
	Argument 2: One of the main categories (abbr, desc, enty, hum, loc, or num)
	Argument 3: Train (or) Test sub-folder (sub_categories/ for train, sub_categories_test/ for test)

	Example: python sub_cate_sep_test.py train_5500.txt abbr sub_categories/
	
The above example groups the subclasses of abbreviation class into different files.

2. Training the Convolution Neural Network.

Since we conducted three sets of experiments (including multi-channel), there would be two files for each word representation model. Execution instructions for word2vec model are given here, and these are same for GloVe and multi-channel model.

There are somethings that you need to download on your system before running this part.

1) word2vec pre-trained model: Visit this link (https://github.com/3Top/word2vec-api) and click on Google News to download the 1.2M pre-trained word2vec model. This should be around 3 GB (after extraction)
2) GloVe: For the single channel GloVe part, we use 300-d representations. Go to this link (https://github.com/3Top/word2vec-api) again, and download Wikipedia + Gigaword 5. Click on 300 dimensional one, which gives a large file with 50, 100, 200, and 300-d vectors. Keep those since 100-d file is used for the multi-channel procedure.
3) GloVe: For the multi-channel part, we also use the vectors learned from twitter data set. Go to this link (https://github.com/3Top/word2vec-api) again and download Twitter (2B tweets). Click on 100 dimensional one.

Before running this code, make sure that you completed the data separation part.
Things to have installed in your system to run this code.
1) Gensim – The library helps in reading the word2vec file.
2) Keras library – This is used to build the neural network models.
3) Theano – To act as a backend to Keras. (Tensorflow can also be used as the backend, but all our programs were only tested on Theano). To change this, please go to keras.json file and change the backend to “theano”.

Primary Network:
To train the network on the first layer categories, you have to run layer_1_cnn_word2vec.py. This takes only one argument.

	Argument 1: Number of Epochs

	Example: python layer_1_cnn_word2vec.py 10

This above execution trains the CNN model on primary categories using word2vec representations for 10 epochs.
The vectors are loaded, and once the training starts, you can visualize the performance of the network on the training data. After training for n epochs, the model is saved and is automatically tested. It should display number of correct predictions, total number of questions, and a confusion matrix.
 
Secondary Networks:
To train the network on the second layer categories using word2vec, you have to run layer_2_cnn_word2vec.py. This takes two arguments.

	Argument 1: Category (abbr, desc, enty, num, hum, or loc)
	Argument 2: Number of Epochs

	Example: python layer_2_cnn_word2vec.py desc 10
This above execution trains the CNN model on sub categories on entity using word2vec representations for 10 epochs.
The vectors are loaded, and once the training starts, you can visualize the performance of the network on the training data. After training for n epochs, the models are saved and are automatically tested. It should display number of correct predictions, total number of questions, and a confusion matrix.

 
The procedure is same for GloVe as well as multi-channel models.

GloVe:

Primary Categories:
	Example: python layer_1_cnn_glove.py 10

This above execution trains the CNN model on primary categories using GloVe representations for 10 epochs.

The vectors are loaded, and once the training starts, you can visualize the performance of the network on the training data. After training for n epochs, the model is saved and is automatically tested. It should display number of correct predictions, total number of questions, and a confusion matrix.

Secondary Categories:
	Example: python layer_2_cnn_glove.py num 10

This above execution trains the CNN model on sub categories of numeric using GloVe representations for 10 epochs. This also saves the models and returns accuracies and confusion matrix.


Multi-channel Network: (Prototype)
We only ran a few tests on this model. Therefore, there is a slight chance that this might throw up some errors.  Since this part is only an idea for the scope of the project, we are not sure about the results.

Primary Categories:
	Example: python layer_1_cnn_multichannel.py 10

This above execution trains the CNN model on primary categories using GloVe 100-d Wiki trained representations and 100-d twitter trained representations for 10 epochs.

The vectors are loaded, and once the training starts, you can visualize the performance of the network on the training data. After training for n epochs, the model is saved and is automatically tested. It should display number of correct predictions, total number of questions, and a confusion matrix.

Secondary Categories:
	Example: python layer_2_cnn_multichannel.py loc 10

This above execution trains the CNN model on sub categories of location using pre-trained GloVe wiki  and GloVe twitter data trained representations for 10 epochs. This also saves the models and returns accuracies and confusion matrix.

3. Testing Pre-Trained Models:
	
	To download the pre-trained models, please go to this link. (Will be updated)
	
We provide two datasets for testing. UIUC’s TREC 500 data set and 115 questions Quora data set. 

The entire pre-trained set is upto 200 MB, which should be downloaded.

Make sure that you have these installed in your system.
1) Gensim : Library for reading word2vec models
2) Keras: To read the .h5 models.
3) Theano : To act as a backend to Keras. (Tensorflow can also be used as the backend, but all our programs were only tested on Theano). To change this, please go to keras.json file and change the backend to “theano”.

Make sure you download the Google News word2vec pre-trained model file and paste in this directory.

The program connected_model_test_2.py takes the test file as argument.

	Argument 1: Test data file in the format shown above.
			(test_data.txt for TREC data set,
			quora_test_set.txt for Quora data set)
This should return the prediction for each question along with the confidence value of its corresponding prediction.

	Example: python connected_model_test_2.py  test_data.txt

The above line executes the connected model code, which is generally an amalgamation of all the models positioned according to the prediction flow. This should return the predicted primary and secondary categories of each example along with a confidence value. At the end, this shows the total number of questions, number of correct predictions for primary categories, number of correct predictions of secondary categories, and a confusion matrix for the primary categories.

4. If you want to test your models on the test sets, please used connected_model_test_2.py and change the names of the models in the program with your models. Running it is same as step 3.

