# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 13:13:01 2020

@author: Nathan
"""
import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

def tokenize(data):
    data = data.lower() # To reduce possible characters
    data = re.sub('[^0-9a-zA-Z:,!? ]+', '', data)
    tokenizer = RegexpTokenizer(r'\w+')
    print("Tokenizing...")
    tokens = tokenizer.tokenize(data)
    print("Input tokenized ", len(tokens), " found")
    # Remove all tokens containing stopwords
    #filtered = filter(lambda tok: tok not in stopwords.words('english'), tokens)
    print("Tokens filtered")
    return " ".join(tokens)


file = open("D:\\Datasets\\writingPrompts\\test.wp_target", encoding="utf-8").read()
file = file.lower()
file = re.sub('[^0-9a-zA-Z:,.!? ]+', '', file)
#file = tokenize(file)
files = 20
for i in range(1,files):
    print(i)
    start = int((i-1) * len(file) / files)
    end = int(i * len(file) / files)
    while file[start] != " ":
        start+=1
    while file[end] != " ":
        end+=1
    w = open("wp_" + str(i) + ".txt", "w", encoding="utf-8").write(file[start:end])
    