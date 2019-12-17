# -*- coding: utf-8 -*-

from data_utils import wordcloud

import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#%%
def undersampling(df):
    positive, negative = df.opinion.value_counts()
    df_positive = df[df.opinion == 'positive']
    df_positive = df_positive.sample(negative, random_state=42)
    df_negative = df[df.opinion == 'negative']
    df = pd.concat([df_positive, df_negative])
    df = df.sample(frac=1)
    return df


def sentiment_analysis_data_preparation(df):
    df.drop(df[df.opinion == 'neutral'].index, inplace=True)
    undersampled = undersampling(df)
    return undersampled

'''
def data_preparation(df):
    df = remove_cols(df)
    #Create new feature "opinion" based on vote
    df = vote_to_opinion(df)    
    df.drop(df[df.opinion == 'neutral'].index, inplace=True)
    df = undersampling(df)
    #alternative: df[df['opinion'].map(lambda x: str(x)!="neutral")]
    reviews = df['reviewText'].tolist()
    preprocessed = preprocessing(reviews)
    df['preprocessedReview'] = [' '.join(review) for review in preprocessed]
    return df
'''


def retrieve_opinion(df, sentiment):
    opinion = df[df['opinion'] == sentiment]
    reviews = opinion['preprocessedReview'].tolist()
    wordcloud(reviews)
    

def vectorization(df, cvector):
    cvector.fit(df.preprocessedReview)
    
    negative_matrix = cvector.transform(df[df['opinion'] == 'negative']['preprocessedReview'])
    negative_words = negative_matrix.sum(axis=0)
    negative_frequency = [(word, negative_words[0, idx]) for word, idx in cvector.vocabulary_.items()]
    negative_tf = pd.DataFrame(list(sorted(negative_frequency, key = lambda x: x[1], reverse=True)),
                               columns=['Terms','negative'])
    negative_tf = negative_tf.set_index('Terms')
    
    positive_matrix = cvector.transform(df[df['opinion'] == 'positive']['preprocessedReview'])
    positive_words = positive_matrix.sum(axis=0)
    positive_frequency = [(word, positive_words[0, idx]) for word, idx in cvector.vocabulary_.items()]
    positive_tf = pd.DataFrame(list(sorted(positive_frequency, key = lambda x: x[1], reverse=True)),
                               columns=['Terms','positive'])
    positive_tf = positive_tf.set_index('Terms')
    
    term_frequency_df = pd.concat([negative_tf, positive_tf], axis=1)
    term_frequency_df['total'] = term_frequency_df['negative'] + term_frequency_df['positive']
    return term_frequency_df


def plot_frequency(df):
    #Frequency plot
    y_pos = np.arange(500)
    plt.figure(figsize=(10,8))
    s = 1
    expected_zipf = [df.sort_values(by='total', ascending=False)['total'][0]/(i+1)**s for i in y_pos]
    plt.bar(y_pos, df.sort_values(by='total', ascending=False)['total'][:500], align='center', alpha=0.5)
    plt.plot(y_pos, expected_zipf, color='r', linestyle='--', linewidth=2, alpha=0.5)
    plt.ylabel('Frequency')
    plt.title('Top 500 tokens in reviews')


def zipf_law(df):
    # Plot of absolute frequency
    from pylab import arange, argsort, loglog, logspace, log10, text
    counts = df.total
    tokens = df.index
    ranks = arange(1, len(counts)+1)
    indices = argsort(-counts)
    frequencies = counts[indices]
    plt.figure(figsize=(8,6))
    plt.ylim(1,10**6)
    plt.xlim(1,10**6)
    loglog(ranks, frequencies, marker=".")
    plt.plot([1,frequencies[0]],[frequencies[0],1],color='r')
    plt.title("Zipf plot for phrases tokens")
    plt.xlabel("Frequency rank of token")
    plt.ylabel("Absolute frequency of token")
    plt.grid(True)
    for n in list(logspace(-0.5, log10(len(counts)-2), 15).astype(int)):
        dummy = text(ranks[n], frequencies[n], " " + tokens[indices[n]], 
                     verticalalignment="bottom",
                     horizontalalignment="left")


def token_frequency(df, sentiment):
    y_pos = np.arange(50)
    plt.figure(figsize=(12,10))
    plt.bar(y_pos, df.sort_values(by=sentiment, ascending=False)[sentiment][:50], align='center', alpha=0.5)
    plt.xticks(y_pos, df.sort_values(by=sentiment, ascending=False)[sentiment][:50].index,rotation='vertical')
    plt.ylabel('Frequency')
    plt.xlabel('Token')
    plt.title('Top 50 tokens in {} reviews'.format(sentiment))
    