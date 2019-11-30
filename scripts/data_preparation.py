# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style("dark")

from data_utils import *
from collections import Counter
import os

#%%
# You must be in \reviews-sentiment folder
os.chdir("..")

# Load dataset
path = r'.\datasets\Grocery_and_Gourmet_Food_5.json'
df = load_dataset(path)
#Features selection
df = remove_cols(df)
#Create new feature "opinion" based on vote
df = vote_to_opinion(df)

#%%

reviews_values = df.reviewText.values
reviews = [reviews_values[i] for i in range(len(reviews_values))]
#%%
#tokenized_reviews = tokenization(reviews)
filtered_reviews = removing_stop_words(reviews)

#%%
reviews = tokenization(filtered_reviews)
#%%

reviews = df.reviewText.values
labels = df.opinion.values

if df.opinion[3] == "positive":
    print("\nPositive:", reviews[3][:90], "...")
else:
    print("\nNegative:", reviews[3][:90], "...")
    
positive_reviews = [reviews[i] for i in range(len(reviews)) if labels[i] == "positive"]
negative_reviews = [reviews[i] for i in range(len(reviews)) if labels[i] == "negative"]

#%% Tokenization

positive_tokenized = tokenization(positive_reviews)
negative_tokenized = tokenization(negative_reviews)

#%% Stop words

positive_filtered = removing_stop_word(positive_tokenized)
negative_filtered = removing_stop_words(negative_tokenized)

