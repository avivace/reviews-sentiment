# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import re

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


#%% Data preprocessing
 
contractions_dict = { 
"ain't": "am not / are not / is not / has not / have not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"I'd": "I had / I would",
"I'd've": "I would have",
"I'll": "I shall / I will",
"I'll've": "I shall have / I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}

contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
punctuation_re = re.compile('([!,.:;?])(\w)')

def expand_contractions(string, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, string)


def fix_punctuation(string, contractions_dict=contractions_dict):
    def replace(match):
        print(match)
        print(match.group(1) + ' ' + match.group(2))
        return match.group(1) + ' ' + match.group(2)
    return punctuation_re.sub(replace, string)

def removing_stop_words(reviews):
    stopwords = nltk.corpus.stopwords.words("english")
    stopwords.remove('not')
    stopwords.remove('and')
    stopwords.remove('or')
    stopwords.remove('but')
    filtered_reviews = []
    for review in reviews:
        review = fix_punctuation(review)
        review = expand_contractions(review)
        filtered_review = (' '.join([word for word in review.split() if word not in stopwords]))
        filtered_review = str(filtered_review)
        filtered_review = re.sub(r'\(.*?\)','', filtered_review)
        filtered_reviews.append(filtered_review)
    return filtered_reviews



def old_tokenization(reviews_list):
    tokenized_reviews = []
    for review in reviews_list:
        review_tokenized = []
        for word in word_tokenize(review):
            review_tokenized.append(word)
        tokenized_reviews.append(review_tokenized)
    return tokenized_reviews

#%%
def pos_tagging(df):
    
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

