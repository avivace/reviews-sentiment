# -*- coding: utf-8 -*-

### Import libraries ###
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
nltk.download('wordnet')
import gensim
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pyLDAvis.gensim


import os

from topic_sentiment_data_preparation import bag_of_words
from topic_sentiment_data_preparation import create_dictionary
from topic_sentiment_data_preparation import tf_idf
from topic_sentiment_data_preparation import data_preparation
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
    fig, ax0 = plt.subplots()
    ax0.plot(x_axis, coherence)
    ax0.set_xlabel("Number of topics")
    ax0.set_ylabel("Coherence score")
    ax0.figure.savefig('figures/3_coherence_plot.svg', format='svg')
    
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
    pyLDAvis.save_html(lda_display, 'figures/lda.html')
    
    
def format_topics_sentences(model, corpus, texts):
    # Init output
    df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(model[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = model.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                df = df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    #df.columns = ['Dominant_Topic', 'Topic_Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    df = pd.concat([df, contents], axis=1)
    df = df.reset_index()
    df.columns = ['review', 'topic_num', 'topic_perc_contribution', 'keywords', 'text']
    return df


def sentiment_polarity(df):
    sentiment = pd.DataFrame()
    sentiment = pd.concat([sentiment, df], ignore_index=True)
    analyser = SentimentIntensityAnalyzer()
    sentiment['sentiments'] = sentiment['text'].str.join(' ').apply(lambda x:
                                                          analyser.polarity_scores(x))
    sentiment = pd.concat([sentiment.drop(['sentiments'], axis=1), 
                           sentiment['sentiments'].apply(pd.Series)],
                          axis=1)
    # Numbers of words
    sentiment['words_nb'] = sentiment["text"].apply(lambda x: len(x))
    sentiment_final = sentiment.groupby(['topic_num', 
                                         'keywords']).agg({'neg':'mean',
                                                           'neu':'mean',
                                                           'pos':'mean',
                                                           'compound':'mean',
                                                           'topic_perc_contribution':'count'}).reset_index()
    return sentiment_final, sentiment


def most_representative_document(df):

    # Most representative document for each topic
    sent_topics_sorted_df = pd.DataFrame()
    
    sent_topics_outdf_grpd = df.groupby('topic_num')
    
    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorted_df = pd.concat([sent_topics_sorted_df, 
                                           grp.sort_values(['topic_perc_contribution'], 
                                                           ascending=[0]).head(1)], 
                                           axis=0)
    sent_topics_sorted_df.reset_index(drop=True, inplace=True)
    sent_topics_sorted_df.columns = ['review', 'topic_num', 'topic_perc_contribution', 'keywords', 'text']
    sent_topics_sorted_df.drop(['review'], axis=1, inplace=True)
    return sent_topics_sorted_df


def topic_distribution_across_documents(df, sentiment):
    # Number of Documents for Each Topic
    sentiment.rename(columns={'dominant_topic':'topic'})
    topic_counts = df['topic_num'].value_counts()
    topic_contribution = round(topic_counts/topic_counts.sum(), 4)
    topic_contribution.rename(columns={'topic_num':'perc_contribution'})
    df_dominant_topics = pd.concat([sentiment, topic_contribution], axis=1)
    
    # Change Column names
    #df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']
    
    # Show
    return df_dominant_topics


    
def run(df):
    current_directory = os.getcwd()
    os.chdir('..')
    preprocessed = data_preparation(df)
    bigram_reviews = make_bigrams(preprocessed)
    dictionary = create_dictionary(bigram_reviews)
    max_topics = 10
    
    # LDA using Bag of Words
    bow_corpus = bag_of_words(bigram_reviews, dictionary)
    #alpha = list(np.arange(0.01, 1, 0.3))
    #beta = list(np.arange(0.01, 1, 0.3))
    alpha = [1]
    beta = [0.1]
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


    # Alcune probs sono 0.5, valutare se togliere quelle comprese tra 0.4 e 0.6
    topic_sents_keywords = format_topics_sentences(best_model, bow_corpus, bigram_reviews)

    topic_sents_keywords.to_pickle('dataframes/topic_sents_keywords.pkl')
    
    sentiment_df, words = sentiment_polarity(topic_sents_keywords)
    pos = words[words["words_nb"] >= 5].sort_values("pos", ascending = False)[["text", "pos"]].head(20)
    neg = words[words["words_nb"] >= 5].sort_values("neg", ascending = False)[["text", "neg"]].head(20)
    
    pos.to_pickle('dataframes/positive.pkl')
    
    neg.to_pickle('dataframes/negative.pkl')
        
    most_repr_rews = most_representative_document(topic_sents_keywords)

    
    most_repr_rews.to_pickle('dataframes/most_repr_rews.pkl')
    
    df_dominant_topics = topic_distribution_across_documents(topic_sents_keywords, sentiment_df)
    
    df_dominant_topics.to_pickle('dataframes/dominant_topics.pkl')

    os.chdir(current_directory)
