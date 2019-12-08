# -*- coding: utf-8 -*-

from data_utils import preprocessing, wordcloud
from data_exploration import most_reviewed_products
import gensim
from gensim import models

from collections import defaultdict

def remove_less_frequent_words(reviews):
    frequency = defaultdict(int)
    for review in reviews:
        for token in review:
            frequency[token] += 1
    
    cleaned = [[token for token in review if frequency[token] > 1] for review in reviews]
    return cleaned
            
def make_bigrams(texts, bigram):
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in texts]
    
def preprocessing_reviews(df):
    # Most frequent product 
    product_id = df.asin.mode().iloc[0]
    # Create a dataframe composed by the most reviewed product
    df_product = df[df['asin'] == product_id]
    reviews = df_product['reviewText'].tolist()
    preprocessed = preprocessing(reviews)
    cleaned = remove_less_frequent_words(preprocessed)
    bigram = gensim.models.Phrases(cleaned, min_count=5, threshold=100)
    wordcloud(cleaned)
    return cleaned, bigram


def create_dictionary(df, texts):
    dictionary = gensim.corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n=100000)
    return dictionary
    

def bag_of_words(df, texts):
    dictionary = create_dictionary(df, texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return corpus
    

def tf_idf(df, corpus):
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    return corpus_tfidf