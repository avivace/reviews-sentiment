# -*- coding: utf-8 -*-

from data_utils import preprocessing, wordcloud
from data_exploration import most_reviewed_products
import gensim
from gensim import models

from collections import defaultdict


def preprocessing_reviews_top_products(df, top_products):
    df_products = most_reviewed_products(df, top_products)
    reviews = df_products['reviewText'].tolist()
    preprocessed_reviews = preprocessing(reviews)
    wordcloud(preprocessed_reviews)
    return preprocessed_reviews

def remove_less_frequent_words(reviews):
    frequency = defaultdict(int)
    for review in reviews:
        for token in review:
            frequency[token] += 1
    
    cleaned_reviews = [[token for token in review if frequency[token] > 1] for review in reviews]
    return cleaned_reviews
            
    
def preprocessing_reviews(df):
    # Most frequent product 
    product_id = df.asin.mode().iloc[0]
    # Create a dataframe composed by the most reviewed product
    df_product = df[df['asin'] == product_id]
    reviews = df_product['reviewText'].tolist()
    preprocessed_reviews = preprocessing(reviews)
    cleaned_reviews = remove_less_frequent_words(preprocessed_reviews)
    wordcloud(cleaned_reviews)
    return cleaned_reviews


def create_dictionary(df, preprocessed_reviews):
    dictionary = gensim.corpora.Dictionary(preprocessed_reviews)
    dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n=100000)
    return dictionary
    

def bag_of_words(df, preprocessed_reviews):
    dictionary = create_dictionary(df, preprocessed_reviews)
    bow_corpus = [dictionary.doc2bow(text) for text in preprocessed_reviews]
    return bow_corpus
    

def tf_idf(df, bow_corpus):
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    return corpus_tfidf