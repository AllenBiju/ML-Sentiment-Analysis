from nltk.tokenize import word_tokenize

def find_feat(document,words):
    word=word_tokenize(document)
    feat={}
    for w in words:
        feat[w]=(w in word)

    return feat