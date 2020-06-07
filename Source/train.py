import random
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC,LinearSVC,NuSVC

import pickle
from nltk.classify import ClassifierI
from statistics import mode
import re
from VoterClassifer import voteclassifier
from find_features import find_feat

stop_words=set(stopwords.words("english"))


with open("C:\\Users\\Allen Biju Thomas\\Desktop\\NEWSPAPER\\Source\\labeled_data.pickle",'rb') as dic:
     labeled_data=pickle.load(dic)
     
# doc=[]
# pos=[]
# neg=[]
# pos_words=[]
# neg_words=[]

# for i in business:
#     if business[i]==1:
#         doc.append((i,"pos"))
#         pos.append(i)
#     elif business[i]==-1:
#         doc.append((i,"neg"))
#         neg.append(i)

all_words=[]

# for i in doc:
for i in labeled_data:
    a=word_tokenize(i)
    for word in a:
        word=word.lower()  
        if word not in stop_words:
            if word not in string.punctuation:
                if word.isalpha():
                    all_words.append(word)


all_words=nltk.FreqDist(all_words)
#words=(all_words.most_common(10000).keys())
words=[i[0] for i in all_words.most_common(5000)]
print("\nList of all words(first 30):\n",words[0:30],".......\n")



# feature_Sets=[(find_feat(rev,words),category) for (rev,category) in doc]
feature_Sets=[(find_feat(rev,words),"pos" if category==1 else "neg" ) for (rev,category) in labeled_data.items()]
random.shuffle(feature_Sets)


train=feature_Sets[:900]
test=feature_Sets[900:]


classifier=nltk.NaiveBayesClassifier.train(train)
print("\noriginal naive bayes accuracy:",nltk.classify.accuracy(classifier,test))
#classifier.show_most_informative_features(15)

mnb_classifier=SklearnClassifier(MultinomialNB())
mnb_classifier.train(train)
print("\nmnb_classifier accuracy:",nltk.classify.accuracy(mnb_classifier,test))


bnb_classifier=SklearnClassifier(BernoulliNB())
bnb_classifier.train(train)
print("\nbnb_classifier accuracy:",nltk.classify.accuracy(bnb_classifier,test))

logr_classifier=SklearnClassifier(LogisticRegression())
logr_classifier.train(train)
print("\nlogr_classifier accuracy:",nltk.classify.accuracy(logr_classifier,test))


sgd_classifier=SklearnClassifier(SGDClassifier())
sgd_classifier.train(train)
print("\nsgd_classifier accuracy:",nltk.classify.accuracy(sgd_classifier,test))


linearsvc_classifier=SklearnClassifier(LinearSVC())
linearsvc_classifier.train(train)
print("\nlinearsvc_classifier accuracy:",nltk.classify.accuracy(linearsvc_classifier,test))


svc_classifier=SklearnClassifier(SVC())
svc_classifier.train(train)
print("\nsvc_classifier accuracy: ",nltk.classify.accuracy(svc_classifier,test))


voteclass=voteclassifier(classifier,
                         mnb_classifier,
                         bnb_classifier,
                         linearsvc_classifier,
                         sgd_classifier,
                         svc_classifier,
                         logr_classifier)


print("\nAccuracy of The model: ",nltk.classify.accuracy(voteclass,test))                     

# save_dict = {"classifier":voteclass}

with open("C:\\Users\\Allen Biju Thomas\\Desktop\\NEWSPAPER\\Source\\ensemble_model.pickle",'wb') as model_file:
#    print("file opened",model_file)
    pickle.dump({
        "classifier":voteclass,
        "words":words
        },model_file)

#print("\n....Model Saved as ensemble_model.pickle")
print("Model Saved saved at: ",os.path.abspath("ensemble_model.pickle"))
