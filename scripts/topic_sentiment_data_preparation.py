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

    
def preprocessing_reviews(df):
    # Most frequent product 
    product_id = df.asin.mode().iloc[0]
    # Create a dataframe composed by the most reviewed product
    df_product = df[df['asin'] == product_id]
    reviews = df_product['reviewText'].tolist()
    preprocessed = preprocessing(reviews)
    cleaned = remove_less_frequent_words(preprocessed)
    # Remove empty lists
    cleaned = [e for e in cleaned if e]
    wordcloud(cleaned)
    return cleaned


def create_dictionary(texts):
    dictionary = gensim.corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n=100000)
    dictionary.compactify()
    return dictionary
    
            
def make_bigrams(texts):
    bigram = gensim.models.Phrases(texts, min_count=5, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in texts]


def bag_of_words(texts):
    dictionary = create_dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return corpus
    

def tf_idf(corpus):
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    return corpus_tfidf