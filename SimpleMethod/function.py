from pythainlp.util import *
from pythainlp.tokenize import word_tokenize

def NGramModel(dataset, foodDict):
    hit = 0
    foodName = []

    for sentence in dataset:
        match = 0
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

    return hit, foodName


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
    f = open("output.txt", "w+", encoding="utf-8")
    for x in range(0,len(dataset)):
        line = dataset[x] + ',' + foodName[x] + ',' + opinion[x] +'\n'
        f.write(line)
    f.close()