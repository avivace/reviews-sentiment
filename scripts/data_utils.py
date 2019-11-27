# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import word_tokenize 
nltk.download('punkt')
nltk.download('stopwords')


#%% Basic functions
def load_dataset(pathfile):
    df = pd.read_json(pathfile, lines=True)
    df['vote'].fillna(0, inplace=True)
    df['vote'] = pd.to_numeric(df['vote'], errors='coerce')
    df.dropna(inplace=True) 
    return df
    
def remove_cols(df):
    df = df.drop(['image',
              'reviewTime',
              'reviewerID',
              'reviewerName',
              'style',
              'unixReviewTime'], axis=1)
    return df
    
def vote_to_opinion(df):
    df.loc[df.overall == 3, 'opinion'] = "neutral"
    df.loc[df.overall > 3, 'opinion'] = "positive"
    df.loc[df.overall < 3, 'opinion'] = "negative"
    return df

#%% Exploration and analysis
def most_reviewed_products(df, n_products):
    reviews_per_product = df['asin'].value_counts()
    most_reviews = reviews_per_product.nlargest(n_products)
    most_reviews = most_reviews.reset_index()
    most_reviews = most_reviews.drop('asin', axis=1)
    
    definitive = df.merge(most_reviews, left_on='asin', right_on='index')
    definitive = definitive.drop('index', axis=1)
    
    return definitive

def significative_reviews(df, n_votes):
    return df[df['vote'] >= n_votes]

#%% Data preprocessing

def tokenization(reviews_list):
    tokenized_reviews = []
    for review in reviews_list:
        review_tokenized = []
        for word in word_tokenize(review):
            review_tokenized.append(word)
        tokenized_reviews.append(review_tokenized)
    return tokenized_reviews


def removing_stop_words(tokenized_reviews):
    stopwords = nltk.corpus.stopwords.words("english")
    stopwords.remove('not')
    stopwords.remove('and')
    stopwords.remove('or')
    stopwords.remove('but')
    filtered_reviews = []
    for review in tokenized_reviews:
        filtered_review = []
        for word in review:
            if word not in stopwords:
                filtered_review.append(word)
        filtered_reviews.append(filtered_review)
    return filtered_reviews

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

def lemmatization(filtered_reviews):
    pass

