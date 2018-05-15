from pythainlp.tokenize import word_tokenize
from pythainlp.sentiment import sentiment
import csv
import re
from function import NGramModel

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
        opinion.append(sentiment(x))
        match = 0
        for word in word_tokenize(x, engine='newmm'):
            if word in foodDict:
                foodName.append(word)
                hit += 1
                match = 1
                break
        if match == 0:
            foodName.append("None")
    return hit

def writeOutput():
    f = open("output.txt", "w+", encoding="utf-8")
    for x in range(0,len(dataset)):
        line = dataset[x] + ',' + foodName[x] + ',' + opinion[x] +'\n'
        f.write(line)
    f.close()

# Main code goes here

# immutable variables 
datasetPath = "../Dataset/Raw_Data.csv"
dictionaryPath = "../Dataset/Dictionary.csv"

# mutable variables

foodDict = readDictionary(dictionaryPath)
dataset, row = readDataset(datasetPath)
foodName = []
opinion = []

print('Load {} rows successfully'.format(row))

#hit = trainSimpleModel()
hit = NGramModel(dataset, foodDict)
#writeOutput()

print('Hit: ' + str(hit))
print('Miss: ' + str(row-hit))
print('Accuracy: {0:.2f}%'.format(hit/row*100))
