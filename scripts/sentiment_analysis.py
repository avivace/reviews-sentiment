# -*- coding: utf-8 -*-
from sentiment_data_preparation import data_preparation
from sentiment_data_preparation import retrieve_opinion
from sentiment_data_preparation import vectorization
from sentiment_data_preparation import plot_frequency
from sentiment_data_preparation import zipf_law
from sentiment_data_preparation import token_frequency

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split   

def run(df):
    df = data_preparation(df)
    retrieve_opinion(df, 'positive')
    retrieve_opinion(df, 'negative')
    term_frequency = vectorization(df)   
    plot_frequency(term_frequency)
    zipf_law(term_frequency)
    token_frequency(term_frequency, 'positive')
    token_frequency(term_frequency, 'negative')
    
    '''
    # Machine learning
    reviews = np.array(df['preprocessedReview'])
    sentiments = np.array(df['opinion']) 
    x_train, y_train, x_test, y_test = train_test_split(reviews, 
                                                        sentiments, 
                                                        test_size=0.2, 
                                                        random_state=42)
    cv = CountVectorizer(stop_words='english',max_features=10000)
    cv_train_features = cv.fit_transform(x_train)
    cv_test_features = cv.transform(x_test)
    
    tv = TfidfVectorizer(min_df=0.0, max_df=1.0, ngram_range=(1,2),
                     sublinear_tf=True)
    tv_train_features = tv.fit_transform(y_train)
    tv_test_features = tv.transform(y_test)
    
    '''