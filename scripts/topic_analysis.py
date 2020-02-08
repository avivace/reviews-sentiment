# -*- coding: utf-8 -*-

### Import libraries ###

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
nltk.download('wordnet')
import gensim
import matplotlib.pyplot as plt
import pandas as pd
import pyLDAvis.gensim
from data_utils import most_reviewed_products
from pathlib import Path

figures_folder = Path("../figures/")
dataframes_folder = Path("../dataframes/")

### Functions ###

def worst_products_asin(df, n_worst):
    if n_worst == 0:
        return []
    top_products = most_reviewed_products(df, 20)
    overall_mean = top_products.groupby(['asin'], as_index=False)['overall'].mean()
    overall_mean = overall_mean.sort_values('overall', ascending=True)
    worst_n_products = overall_mean['asin'].iloc[:n_worst].tolist()
    return worst_n_products
    

def best_products_asin(df, n_best):
    if n_best == 0:
        return []
    top_products = most_reviewed_products(df, 20)
    overall_mean = top_products.groupby(['asin'], as_index=False)['overall'].mean()
    overall_mean = overall_mean.sort_values('overall', ascending=False)
    best_n_products = overall_mean['asin'].iloc[:n_best].tolist()
    return best_n_products


def products_to_analyze(df, n_best=0, n_worst=0):
    worst = worst_products_asin(df, n_worst)
    best = best_products_asin(df, n_best)
    products = worst + best
    if products == []:
        # Most reviewed product
        product_id = df.asin.mode().iloc[0]
        return [product_id]
    else:
        return products
        
    
def create_dictionary(texts):
    dictionary = gensim.corpora.Dictionary(texts)
    #dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n=100000)
    dictionary.filter_extremes(keep_n=10000)
    dictionary.compactify()
    return dictionary
    
            
def make_bigrams(texts):
    bigram = gensim.models.Phrases(texts, min_count=5, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in texts]


def bag_of_words(texts, dictionary):
    corpus = [dictionary.doc2bow(text) for text in texts]
    return corpus
    

def compute_lda_model(corpus, num_topics, dictionary, texts, alpha, beta):
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
    return coherences, lda_models


def compute_multiple_lda_models(alphas, betas, num_topics, corpus, texts, dictionary):
    all_coherences, all_lda_models, all_parameters = [], [], []
    for alpha in alphas:
        for beta in betas:
            coherences, lda_models = compute_lda_model(corpus=corpus, 
                                                       num_topics=num_topics, 
                                                       dictionary=dictionary, 
                                                       texts=texts,
                                                       alpha=alpha,
                                                       beta=beta)
            all_coherences.append(coherences)
            all_lda_models.append(lda_models)
            all_parameters.append([alpha, beta])
    return all_coherences, all_lda_models, all_parameters


def plot_coherence(num_topics, coherence, product_asin):
    x_axis = range(2, 2+num_topics)
    fig, ax0 = plt.subplots()
    ax0.plot(x_axis, coherence)
    ax0.set_xlabel("Number of topics")
    ax0.set_ylabel("Coherence score")
    ax0.figure.savefig(figures_folder / '3_coherence_plot_{0}.svg'.format(product_asin), format='svg')
    
    
def show_topics(model, ideal_topics, num_words, product_asin):
    topics = model.show_topics()
    for topic in topics:
        print(topic)
        
    word_dict = {};
    for i in range(ideal_topics):
        words = model.show_topic(i, topn = num_words)
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = [i[0] for i in words]
    topic_df = pd.DataFrame(word_dict)
    topic_df.to_pickle(dataframes_folder / 'topics_{}.pkl'.format(product_asin))
    print(topic_df)
    
    
def topic_visualization(model, corpus, dictionary, product_asin):
    lda_display = pyLDAvis.gensim.prepare(model, corpus, dictionary, sort_topics=True)
    pyLDAvis.save_html(lda_display, 'lda_{0}.html'.format(product_asin))
    
'''
def format_topics_sentences(model, corpus, texts):
    # Get main topic reviews
    # Init output
    df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(model[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                # probability pairs for the most relevant words generated by the topic
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


'''

def run_for_custom_analysis(df):
    print("CUSTOM LDA ANALYSIS!")
    df = df.head(15000)
    product = "PORTABLECHARGERS"
    reviews_product = [r.split(' ') for r in df['preprocessedReview']]
    bigram_reviews = make_bigrams(reviews_product)
    dictionary = create_dictionary(bigram_reviews)
    bow_corpus = bag_of_words(bigram_reviews, dictionary)
    max_topics = 10
    alpha_list = [0.1, 1]
    beta_list = [0.01, 0.1, 1]
    num_topics = list(range(2, max_topics + 1))
    all_coherences, all_lda_models, all_parameters = compute_multiple_lda_models(alphas=alpha_list,
                                                                                 betas=beta_list,
                                                                                 num_topics=num_topics,
                                                                                 corpus=bow_corpus,
                                                                                 texts=bigram_reviews,
                                                                                 dictionary=dictionary)
    # Extract best coherence and index
    best_coherence_value, index_best_value = max((x, (i, j))
                                                 for i, row in enumerate(all_coherences)
                                                 for j, x in enumerate(row))
    best_alpha = all_parameters[index_best_value[0]][0]
    best_beta = all_parameters[index_best_value[0]][1]
    best_model = all_lda_models[index_best_value[0]][index_best_value[1]]
    print('Best model has {} coherence with {} alpha value and {} beta value'.format(best_coherence_value,
                                                                                     best_alpha,
                                                                                     best_beta))
    best_coherences = all_coherences[index_best_value[0]]
    best_num_topics = num_topics[0] + index_best_value[1]
    print('Best num of topics: {}'.format(best_num_topics))
    plot_coherence(len(num_topics), best_coherences, product)
    show_topics(best_model, best_num_topics, 10, product)
    topic_visualization(best_model, bow_corpus, dictionary, product)


def run(df):    
    product_list = products_to_analyze(df, n_best=3, n_worst=3)
    for product in product_list:
        figures_folder = Path("../figures/")
        name_file = '3_coherence_plot_{0}.svg'.format(product)
        path_file = figures_folder / name_file
        if False:
            print('{} already computed.'.format(product))
        else:
            print(product)
            df_product = df[df['asin'] == product]
            reviews_product = [r.split(' ') for r in df_product['preprocessedReview']]
            bigram_reviews = make_bigrams(reviews_product)
            dictionary = create_dictionary(bigram_reviews)
            bow_corpus = bag_of_words(bigram_reviews, dictionary)
            max_topics = 10
            alpha_list = [0.1, 1]
            beta_list = [0.01, 0.1, 1]
            num_topics = list(range(2, max_topics+1))
            all_coherences, all_lda_models, all_parameters = compute_multiple_lda_models(alphas=alpha_list,
                                                                                         betas=beta_list,
                                                                                         num_topics=num_topics,
                                                                                         corpus=bow_corpus,
                                                                                         texts=bigram_reviews,
                                                                                         dictionary=dictionary)
            # Extract best coherence and index 
            best_coherence_value, index_best_value = max((x, (i, j))
                                                         for i, row in enumerate(all_coherences)
                                                         for j, x in enumerate(row))
            best_alpha = all_parameters[index_best_value[0]][0]
            best_beta = all_parameters[index_best_value[0]][1]
            best_model = all_lda_models[index_best_value[0]][index_best_value[1]]
            print('Best model has {} coherence with {} alpha value and {} beta value'.format(best_coherence_value,
                                                                                             best_alpha,
                                                                                             best_beta))
            best_coherences = all_coherences[index_best_value[0]]
            best_num_topics = num_topics[0] + index_best_value[1]
            print('Best num of topics: {}'.format(best_num_topics))
            plot_coherence(len(num_topics), best_coherences, product)
            show_topics(best_model, best_num_topics, 10, product)
            topic_visualization(best_model, bow_corpus, dictionary, product)
            '''
            topic_sents_keywords = format_topics_sentences(best_model, bow_corpus, bigram_reviews)
        
            topic_sents_keywords.to_pickle('dataframes/topic_sents_keywords.pkl')
            
            sentiment_df, words = sentiment_polarity(topic_sents_keywords)
            pos = words[words["words_nb"] >= 5].sort_values("pos", ascending = False)[["text", "pos"]].head(20)
            neg = words[words["words_nb"] >= 5].sort_values("neg", ascending = False)[["text", "neg"]].head(20)

            most_repr_rews = most_representative_document(topic_sents_keywords)
            df_dominant_topics = topic_distribution_across_documents(topic_sents_keywords, sentiment_df)
            '''
            