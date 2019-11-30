# -*- coding: utf-8 -*-

### Import libraries ###
import pandas as pd 
import numpy as np
import nltk
import re

import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style("dark")

from data_utils import load_dataset, remove_cols, vote_to_opinion, removing_stop_words
from collections import Counter
import os

### Functions ###

def topic_based_tokenization(reviews):
    tokenizedReviews = {}
    key = 1
    #stopwords = nltk.corpus.stopwords.words("english")
    regexp = re.compile(r'\?')
    for review in reviews:
        for sentence in nltk.sent_tokenize(review):
            #logic to remove questions and errors
            if regexp.search(sentence):
                print("removed")
            else:
                sentence=re.sub(r'\(.*?\)','',sentence)
                tokenizedReviews[key]=sentence
                key += 1

    for key,value in tokenizedReviews.items():
        print(key,' ',value)
        tokenizedReviews[key]=value
    return tokenizedReviews

#%%
# You must be in \reviews-sentiment folder
os.chdir("..")

# Load dataset
path = r'.\datasets\Grocery_and_Gourmet_Food_5.json'
df = load_dataset(path)
# Features selection
df = remove_cols(df)
# Create new feature "opinion" based on vote
df = vote_to_opinion(df)
# Most frequent product 
product_id = df.asin.mode().iloc[0]
# Extract reviews
df_product = df[df['asin'] == product_id]

#%%
reviews_values = df_product.reviewText.values
reviews = [reviews_values[i] for i in range(len(reviews_values))]

#%%
reviews_filtered = removing_stop_words(reviews)
reviews_tokenized = topic_based_tokenization(reviews_filtered)
