# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 13:13:01 2020

@author: Nathan
"""
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

def tokenize(input):
    input = input.lower() # To reduce possible characters
    tokenizer = RegexpTokenizer(r'\w+')
    print("Tokenizing...")
    tokens = tokenizer.tokenize(input)
    print("Input tokenized ", len(tokens), " found")
    # Remove all tokens containing stopwords
    filtered = filter(lambda tok: tok not in stopwords.words('english'), tokens)
    print("Tokens filtered")
    return " ".join(filtered)


file = open("D:\\Datasets\\writingPrompts\\test.wp_target", encoding="utf-8").read()
file = file.lower()
file = tokenize(file)
for i in range(1,10):

    start = int((i-1) * len(file) / 10)
    end = int(i * len(file) / 10)
    while file[start] != " ":
        start+=1
    while file[end] != " ":
        end+=1
    w = open("wp_" + str(i) + ".txt", "w", encoding="utf-8").write(file[start:end])
    