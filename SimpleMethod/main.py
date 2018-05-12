from pythainlp.tokenize import word_tokenize
import csv
import re

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

def trainSimpleModel():
    hit = 0
    for x in dataset:
        for word in word_tokenize(x, engine='newmm'):
            if word in foodDict:
                hit += 1
                break
    return hit

# Main code goes here

# immutable variables 
datasetPath = "../Dataset/Raw_Data.csv"
dictionaryPath = "../Dataset/Dictionary.csv"

# mutable variables

foodDict = readDictionary(dictionaryPath)
dataset, row = readDataset(datasetPath)

print('Load {} rows successfully'.format(row))

hit = trainSimpleModel()

print('Hit: ' + str(hit))
print('Miss: ' + str(row-hit))
print('Accuracy: {0:.2f}%'.format(hit/row*100))
