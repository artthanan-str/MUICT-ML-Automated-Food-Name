from pythainlp.util import *
from pythainlp.tokenize import word_tokenize
from pythainlp.sentiment import sentiment
from pythainlp.corpus import stopwords
import string
from nltk import NaiveBayesClassifier as nbc
import codecs
from itertools import chain
import time
import pickle

def predictLabel(sentence, vocabulary):

    # load classifier
    f = open('../sentiment/classifier.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()

    featurized_test_sentence =  {i:(i in word_tokenize(sentence)) for i in vocabulary}
    result = classifier.classify(featurized_test_sentence)
    return result

def sentimentTrainModel():
    startTime = time.time()
    # pos.txt
    with codecs.open('../sentiment/data/pos.txt', 'r', "utf-8") as f:
        lines = f.readlines()
        listpos = [e.strip() for e in lines]
        del lines # Release memory
    f.close()
    
    # neg.txt
    with codecs.open('../sentiment/data/neg.txt', 'r', "utf-8") as f:
        lines = f.readlines()
        listneg = [e.strip() for e in lines]
        del lines  # Release memory
    f.close()

    pos1=['pos']*len(listpos)
    neg1=['neg']*len(listneg)
    training_data = list(zip(listpos,pos1)) + list(zip(listneg,neg1))
    
    vocabulary = set(chain(*[word_tokenize(i[0]) for i in training_data]))
    feature_set = [({i:(i in word_tokenize(sentence)) for i in vocabulary},tag) for sentence, tag in training_data]

    classifier = nbc.train(feature_set)

    endTime = time.time()

    print('Total time used to train model: ' + str(endTime-startTime) + ' mins\n')

    # save classifier
    f = open('../sentiment/classifier.pickle', 'wb')
    pickle.dump(classifier, f)
    f.close()

    return classifier, vocabulary

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
