print("loading modules...")
from nltk.tokenize import word_tokenize,sent_tokenize
import pickle
from textblob import TextBlob,Word
from textblob.sentiments import NaiveBayesAnalyzer
from find_features import find_feat
from VoterClassifer import voteclassifier
print("loaded modules")


l=[]
score=0
voteclass = voteclassifier()
words =[]

with open("C:\\Users\\Allen Biju Thomas\\Desktop\\NEWSPAPER\\Source\\ensemble_model.pickle",'rb') as model_file:
     saved_items=pickle.load(model_file)
     voteclass = saved_items["classifier"]
     words = saved_items["words"]


with open("C:\\Users\\Allen Biju Thomas\\Desktop\\NEWSPAPER\\testdata.txt") as file:
#with open("C:\\Users\\Allen Biju Thomas\\Desktop\\NEWSPAPER\\Datasets\\business\\484.txt") as file:
    for read in file:
        a=sent_tokenize(read)
        l.append(a)
        for i in a:
            analysis=TextBlob(i)
            print(i,"(",analysis.sentiment.polarity,")\n")
            score=score+analysis.sentiment.polarity


    textblob_result = ""
    if score>0:
        textblob_result = "Positive"
    else:
        textblob_result = "Negative"
    print("\n\n-------------------------")    
    print("Sentiment Analysis of textblob: ",textblob_result)
    print("-------------------------\n")
    

    l=[j.replace("\n","") for i in l for j in i]
    
    line=""
    for i in l:
        line=line+" "+str(i)

    feat=find_feat(line,words)
    
    pred = voteclass.classify(feat)
    result = pred
    if pred=="pos":
        result = "Positive"
    elif pred=="neg":
        result = "Negative"

    print("\n-------------------------")
    print("Our model's performance\n")
    print("Classification : ",result,"\nConfidence :",voteclass.confidence(feat))
    print("-------------------------\n")

    score=0
    line=""
    l=[]


