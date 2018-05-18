import csv
import re
from function import NGramModel, searchFood, pythaiSentiment, trainCustomSentiment, predict, writeOutput
from pythainlp.rank import rank
from collections import Counter

# Functions go here

def readDictionary(filePath):
    csvFile = []
    with open(filePath, encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            #print(''.join(row))
            csvFile.append(''.join(row))
    return csvFile

def readDataset(filePath):
    csvFile = []
    i = 0
    with open(filePath, encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            #print(''.join(row))
            csvFile.append(''.join(row))
            i += 1
    return csvFile, i              


# Main code goes here

# immutable variables 
datasetPath = "../dataset/Raw_Data.csv"
dictionaryPath = "../dataset/Dictionary.csv"

# mutable variables

foodDict = readDictionary(dictionaryPath)
dataset, row = readDataset(datasetPath)
foodName = []
opinion = []

print('Load {} rows successfully'.format(row))

# Search food name in food dictionary
print('\n Simple Model, search food name in food dictionary word by word\n')
simple_hit = 0
for sentence in dataset:
    (match, name) = searchFood(sentence, foodDict)
    simple_hit += match

print('Hit: ' + str(simple_hit))
print('Miss: ' + str(row-simple_hit))
print('Accuracy: {0:.2f}%'.format(simple_hit/row*100))
print('===========================================================\n')


# Apply N-Gram to simple model
print('Simple Model with n-gram (2 grams to 6 grams)\n')
ngram_hit = 0
ngram_hit, foodName, opinion = NGramModel(dataset, foodDict)
print('Hit: ' + str(ngram_hit))
print('Miss: ' + str(row-ngram_hit))
print('Accuracy: {0:.2f}%'.format(ngram_hit/row*100))
print('Positive: ' + str(opinion.count("pos")))
print('Negative: ' + str(opinion.count("neg")))

writeOutput(dataset, foodName, opinion)

#print(rank(foodName))

# Sentiment Analysis (Model from PyThaiNLP)
sentiment_txt = "อาหารอร่อยมากเลย"
result = pythaiSentiment(sentiment_txt)
print('===========================================================\n')
print('Sentiment Analysis (Model from PyThaiNLP)\n')
print('Sentiment Result: ' + result)



# Our own sentiment model

sentence = 'อาหารอร่อยมากเลย'

#classifier, vocabulary = trainCustomSentiment() # run this code after editting pos or neg dictionary
print('Finished training model\n')

result = predict(sentence)
print('Sentiment Result: ' + result)


#print('Input sentence: ' + sentence)
#print('Result from classifying: ' + result)

#sentiment_txt = "อาหารอร่อยมากเลย"
#result = pythaiSentiment(sentiment_txt)
#print('===========================================================\n')
#print('Sentiment Analysis (Model from PyThaiNLP)\n')
#print('Sentiment Result: ' + result)
