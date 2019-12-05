# -*- coding: utf-8 -*-

from data_utils import preprocessing, wordcloud
import gensim
from gensim import models

def preprocessing_reviews(df):
    # Most frequent product 
    product_id = df.asin.mode().iloc[0]
    # Create a dataframe composed by the most reviewed product
    df_product = df[df['asin'] == product_id]
    reviews = df_product['reviewText'].tolist()
    preprocessed_reviews = preprocessing(reviews)
    wordcloud(preprocessed_reviews)
    return preprocessed_reviews


def create_dictionary(df, preprocessed_reviews):
    dictionary = gensim.corpora.Dictionary(preprocessed_reviews)
    dictionary.filter_extremes(no_below=10) #, no_above=0.5)#, keep_n=100000)
    return dictionary
    

def bag_of_words(df, preprocessed_reviews):
    dictionary = create_dictionary(df, preprocessed_reviews)
    bow_corpus = [dictionary.doc2bow(text) for text in preprocessed_reviews]
    return bow_corpus
    

def tf_idf(df, bow_corpus):
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    return corpus_tfidf