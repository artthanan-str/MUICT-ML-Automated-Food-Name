from pythainlp.util import *
from pythainlp.tokenize import word_tokenize
from pythainlp.sentiment import sentiment
from pythainlp.corpus import stopwords
import string

def pythaiSentiment(sentence):
    stopword = stopwords.words('thai')
    for word in stopword:
        if word in sentence:
            sentence = sentence.replace(word, "") # Remove stopword
    result = sentiment(sentence)
    return result

def NGramModel(dataset, foodDict):
    hit = 0
    foodName = []
    opinion = []

    for sentence in dataset:
        match = 0
        opinion.append(pythaiSentiment(sentence))
        (match, name) = searchFood(sentence, foodDict) # search all words in food dictionary
        #print(str(match) + ': ' + name)
        if(match == 0):
            for i in range(2,6): # loop n-gram from 2 to 6 grams
                temp = ngrams(word_tokenize(sentence, engine='newmm'), i)
                gramList = list(temp)
                
                for line in gramList:
                    string = ''
                    
                    for word in line:
                        string += word
                    
                    if string in foodDict:
                        match = 1
                        name = string
                        break
                
                if match == 1:
                    break

        hit += match
        foodName.append(name)

    return hit, foodName, opinion


def searchFood(sentence, foodDict):
    hit = 0
    name = "None"
    for word in word_tokenize(sentence, engine='newmm'):
        if word in foodDict:
            name = word
            hit = 1
            break

    return hit, name


def writeOutput(dataset, foodName, opinion):
    f = open("output.csv", "w+", encoding="utf-8")
    for x in range(0,len(dataset)):
        line = dataset[x] + ',' + foodName[x] + ',' + opinion[x] +'\n'
        f.write(line)
    f.close()