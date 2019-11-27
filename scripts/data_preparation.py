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
'''
with open('DatasetsPath.txt', 'r') as myfile:
    pathfile = myfile.read()

df = pd.read_csv(pathfile + '\Gourmet_food.csv', index_col=0)
'''

#%%
# You must be in \reviews-sentiment folder
os.chdir("..")

# Load dataset
path = r'.\datasets\Grocery_and_Gourmet_Food_5.json'
df = load_dataset(path)
df = remove_cols(df)
df = vote_to_opinion(df)

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

#%% STRATEGIA DA VALUTARE - BONESI
cnt_positve = Counter()

for row in positive_reviews:
    cnt_positve.update(row.split(" "))
print("Vocabulary size for positive reviews:", len(cnt_positve.keys()))


cnt_negative = Counter()

for row in negative_reviews:
    cnt_negative.update(row.split(" "))
print("Vocabulary size for negative reviews:", len(cnt_negative.keys()))

cnt_total = Counter()
for row in reviews:
    cnt_total.update(row.split(" "))

pos_neg_ratio = Counter()
vocab_pos_neg = (set(cnt_positve.keys())).intersection(set(cnt_negative.keys()))
for word in vocab_pos_neg:
    if cnt_total[word] > 100:
        ratio = cnt_positve[word] / float(cnt_negative[word] + 1)
    if ratio > 1:
        pos_neg_ratio[word] = np.log(ratio)
    else:
        pos_neg_ratio[word] = -np.log(1 / (ratio + 0.01))

positive_dict = {}
for word, cnt in pos_neg_ratio.items():
    if (cnt > 1):
        positive_dict[word] = cnt
