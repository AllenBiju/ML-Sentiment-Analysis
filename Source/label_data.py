
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
import matplotlib
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet
import string
import csv
import sys
import pickle
from textblob import TextBlob,Word
from nltk import ngrams
import re
from textblob.sentiments import NaiveBayesAnalyzer
import os


l1=[]

labeled_data={}

for i in range(1,511):
    if i<10:
        k="0"+"0"+str(i)
    elif i<100:
        k="0"+str(i)
    else:
        k=str(i)
    l1.append(k)


l=[]

count=0
count1=0
count2=0

score=0
folder_path = "../datasets"

for section in ["business","politics","entertainment"]:
    path = folder_path+"/"+section
    file_list = os.listdir(path)
    print(path)
    print(file_list[:3],"......",file_list[-3:])
    for file_no in file_list:
        with open(path+"/"+file_no) as file:
            for read in file:
                a=sent_tokenize(read)
                l.append(a)
                for i in a:
                    analysis=TextBlob(i)
    ##                print(i,analysis.sentiment.polarity)
                    score=score+analysis.sentiment.polarity
    
    
            l=[j.replace("\n","") for i in l for j in i]
            
            line=""
            for i in l:
                line=line+" "+str(i)
    
            if score>0:
                count=count+1
                labeled_data[line]=1
                # business.appendline=1
            elif score<=0:
                count1=count1+1
                labeled_data[line]=-1
            else:
                count2=count2+1
                labeled_data[line]=0
                
            score=0
            line=""
            l=[]


print("\nTotal labeled articles: ",(count+count1))
print("No. of positive articles: ",count,"\nNo. of negative articles:",count1,"\nNo. of unlabeled articles: ",count2)


with open("labeled_data.pickle",'wb') as dic:
     pickle.dump(labeled_data,dic,protocol=pickle.HIGHEST_PROTOCOL)

print("labled data saved at: ",os.path.abspath("labeled_data.pickle")
