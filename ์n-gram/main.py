from pythainlp.util import *
from pythainlp.tokenize import word_tokenize

text='วันนี้ฉันมีนิทานมาเล่านิทานให้เธอฟังเรื่องท.ทหารอดทน'
token = word_tokenize(text, engine='newmm')

#s = bigrams(word_tokenize(text, engine='newmm'))
x = list(ngrams(token, 3))

for i in x:
    print(i)