# -*- coding: utf-8 -*-

### Import libraries ###
import nltk
import gensim
nltk.download('wordnet')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyLDAvis.gensim

import os

from topic_sentiment_data_preparation import bag_of_words
from topic_sentiment_data_preparation import create_dictionary
from topic_sentiment_data_preparation import tf_idf
from topic_sentiment_data_preparation import preprocessing_reviews
from topic_sentiment_data_preparation import make_bigrams

### Functions ###

def evaluate_multiple_lda(corpus, num_topics, dictionary, texts, alpha, beta):
    lda_models, coherences = [], []
    for n in num_topics:
        model = gensim.models.LdaModel(corpus=corpus, 
                                       num_topics=n, 
                                       random_state=42, 
                                       chunksize=100,
                                       id2word=dictionary, 
                                       passes=10, 
                                       alpha=alpha,
                                       eta=beta)
        lda_models.append(model)
        cm = gensim.models.ldamodel.CoherenceModel(model=model, 
                                                   dictionary=dictionary, 
                                                   coherence='c_v', 
                                                   texts=texts)
        coherences.append(cm.get_coherence())
        print('\nNumber of topic:', n)
        for idx, topic in model.print_topics(-1):
            print('\nTopic: {} \nWords: {}'.format(idx, topic)) 
    return coherences, lda_models

def plot_coherence(num_topics, coherence):
    x_axis = range(2, 2+num_topics)
    plt.plot(x_axis, coherence)
    plt.xlabel("Number of topics")
    plt.ylabel("Coherence score")
    plt.show()
    
def show_topics(model, ideal_topics, num_words):
    topics = model.show_topics()
    for topic in topics:
        print(topic)
        
    word_dict = {};
    for i in range(ideal_topics):
        words = model.show_topic(i, topn = num_words)
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = [i[0] for i in words]
    topic_df = pd.DataFrame(word_dict)
    print(topic_df)
    
def topic_visualization(model, corpus, dictionary):
    lda_display = pyLDAvis.gensim.prepare(model, corpus, dictionary, sort_topics=True)
    #pyLDAvis.display(lda_display)
    lda_display
    current_directory = os.getcwd()
    os.chdir('..')
    pyLDAvis.save_html(lda_display, 'figures/lda.html')
    os.chdir(current_directory)

    
def run(df):
    preprocessed = preprocessing_reviews(df)
    bigram_reviews = make_bigrams(preprocessed)
    dictionary = create_dictionary(bigram_reviews)
    max_topics = 10
    
    # LDA using Bag of Words
    bow_corpus = bag_of_words(bigram_reviews)
    alpha = list(np.arange(0.01, 1, 0.3))
    beta = list(np.arange(0.01, 1, 0.3))
    num_topics = list(range(2, max_topics+1))
    all_coherences, all_lda_models, parameters = [], [], []
    for a in alpha:
        for b in beta:
            coherences, lda_models = evaluate_multiple_lda(corpus=bow_corpus, 
                                                           num_topics=num_topics, 
                                                           dictionary=dictionary, 
                                                           texts=bigram_reviews,
                                                           alpha=a,
                                                           beta=b)
            all_coherences.append(coherences)
            all_lda_models.append(lda_models)
            parameters.append([a, b])
            print(coherences)
    print('COHERENCES -----------')
    print(all_coherences)
    # Extract best coherence and index 
    best_coherence_value, index_best_value = max((x, (i, j))
                                                 for i, row in enumerate(all_coherences)
                                                 for j, x in enumerate(row))
    best_alpha = parameters[index_best_value[0]][0]
    best_beta = parameters[index_best_value[0]][1]
    best_model = all_lda_models[index_best_value[0]][index_best_value[1]]
    print('Best model has {} coherence with {} alpha value and {} beta value'.format(best_coherence_value,
                                                                                     best_alpha,
                                                                                     best_beta))
    best_coherences = all_coherences[index_best_value[0]]
    best_num_topics = num_topics[0] + index_best_value[1]
    plot_coherence(num_topics=len(num_topics), coherence=best_coherences)
    show_topics(best_model, best_num_topics, num_words=10)
    topic_visualization(best_model, bow_corpus, dictionary)
    
    '''
    # LDA using Tf-Idf
    corpus_tfidf = tf_idf(bow_corpus)
    coherence_tfidf = evaluate_topic(corpus=corpus_tfidf, num_topics=num_topics, dictionary=dictionary, texts=preprocessed_reviews)
    plot_coherence(num_topics=num_topics, coherence=coherence_tfidf)
    '''
