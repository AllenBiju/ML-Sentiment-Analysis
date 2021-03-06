# Ensemble Learning Sentiment Analysis

<br/>

In this project we create an ensemble learning ML model to predict the sentiment (Positive/Negative) of a news article. 
The dataset used was the raw text files from- http://mlg.ucd.ie/datasets/bbc.html <br/>
There are 3 main files-
- lablel_data.py
- train.py
- test_sentiment.py

<br/>

#### lablel_data.py
This file is used to label the dataset that we use to train our model. Data was collected from the BBC news dataset. This dataset contains articles published by BBC organised into categories like, business, sport, entertainment etc.
In this file we loop through each article and pass it to textblob. Textblob will give us the sentiment polarity for each sentence which will add up to the sentiment polarity value of the article. The article text is stored in a dictionary as the key and the label which we get from the polarity will be the value.
This dictionary will be stored as a .pickle file using the pickle module

#### train.py
This file trains our prediction model with the labelled dataset.
Here we have defined a class voteclassifier that will take 7 trained prediction models as an input.
We extract the previously stored dictionary from the .pickle file. The code will loop through the dictionary and the text of the article will be converted to a vectorised from using the find_feat function. A list is created that contains every article in the from of a tuple of the vectorised text and the label of the article.
This list is now split into the train and test dataset.
Each individual model (NaiveBayesClassifier, MultinomialNB, BernoulliNB, LogisticRegression, SGDClassifier, LinearSVC, SVC) is trained using the training set. The accuracy of each model is also checked using the test data.
All 7 models will be passed to the voteclassifier class. This class will call the classify function for each model and return the mode of all the classifications. This will be the final prediction of our model.

#### test_sentiment.py
This file will show the final prediction for an unseen new article. 
It opens the testdata.txt file and classifies the text of the article and displays the result along with the confident. In order to test this output we will also run the text through the textblob pipeline.
Both outputs will be displayed and can be compared.

