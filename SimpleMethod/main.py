import csv
import re
from function import NGramModel, searchFood, pythaiSentiment, trainCustomSentiment, predict, writeOutput, evaluateSentiment
from pythainlp.rank import rank
from collections import Counter
import time

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
    data, label = [], []
    i = 0
    with open(filePath, encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            #print(''.join(row))
            data.append(row[0])
            label.append(row[2])
            i += 1
    return data, label, i              


# Main code goes here

# immutable variables 
datasetPath = "../dataset/Raw_Data_ReadyToUse.csv"
dictionaryPath = "../dataset/Dictionary.csv"

# mutable variables

foodDict = readDictionary(dictionaryPath)
dataset, label, row = readDataset(datasetPath)
foodName = []
opinion1 = []
opinion2 = []

print('Load {} rows successfully'.format(row))

# Search food name in food dictionary
print('\n Simple Model, search food name in food dictionary word by word\n')
simple_hit = 0
#startTime = time.time()
for sentence in dataset:
    (match, name) = searchFood(sentence, foodDict)
    simple_hit += match
#endTime = time.time()
print('Hit: ' + str(simple_hit))
print('Miss: ' + str(row-simple_hit))
print('Hit Rate: {0:.2f}%'.format(simple_hit/row*100))
#print('Total time used to extract: ' + str(endTime-startTime) + ' seconds\n')
print('===========================================================\n')


# Apply N-Gram to simple model
print('Simple Model with n-gram (2-6 grams)\n')
#trainCustomSentiment()
ngram_hit = 0
#startTime = time.time()
ngram_hit, foodName, opinion1, opinion2 = NGramModel(dataset, foodDict)
#endTime = time.time()
tp1, tn1, fp1, fn1 = evaluateSentiment(label, opinion1)
tp2, tn2, fp2, fn2 = evaluateSentiment(label, opinion2)
accuracy1 = (tp1+tn1)/row*100
accuracy2 = (tp2+tn2)/row*100

print('Hit: ' + str(ngram_hit))
print('Miss: ' + str(row-ngram_hit))
print('Hit Rate: {0:.2f}%'.format(ngram_hit/row*100))
#print('Total time used to extract: ' + str(endTime-startTime) + ' seconds\n')
print('\nPyThai Sentiment Result')
print('Positive: ' + str(tp1+fp1))
print('Negative: ' + str(tn1+fn1))
print('Accuracy: {0:.2f}%'.format(accuracy1))
print('True Positive: ' + str(tp1))
print('True Negative: ' + str(tn1))
print('False Positive: ' + str(fp1))
print('False Negative: ' + str(fn1))
print('\nCustom Sentiment Result')
print('Positive: ' + str(tp2+fp2))
print('Negative: ' + str(tn2+fn2))
print('Accuracy: {0:.2f}%'.format(accuracy2))
print('True Positive: ' + str(tp2))
print('True Negative: ' + str(tn2))
print('False Positive: ' + str(fp2))
print('False Negative: ' + str(fn2))

writeOutput(dataset, foodName, opinion1, "output1.csv")
writeOutput(dataset, foodName, opinion1, "output2.csv")

#print(rank(foodName))
