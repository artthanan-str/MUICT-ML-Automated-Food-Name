from pythainlp.tokenize import word_tokenize
from pythainlp.sentiment import sentiment
import csv
import re
from function import NGramModel, searchFood
from pythainlp.rank import rank

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
ngram_hit, foodName = NGramModel(dataset, foodDict)
print('Hit: ' + str(ngram_hit))
print('Miss: ' + str(row-ngram_hit))
print('Accuracy: {0:.2f}%'.format(ngram_hit/row*100))


#writeOutput()


print(rank(foodName))