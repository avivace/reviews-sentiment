# -*- coding: utf-8 -*-

### Import libraries ###
import nltk
import gensim
nltk.download('wordnet')
import matplotlib.pyplot as plt
import pandas as pd
import pyLDAvis.gensim

from topic_sentiment_data_preparation import bag_of_words, create_dictionary, tf_idf, preprocessing_reviews, make_bigrams

### Functions ###

def evaluate_multiple_lda(corpus, num_topics, dictionary, texts):
    lda_model, coherence_list = [], []
    for n in range(1, num_topics):
        model = gensim.models.LdaModel(corpus=corpus, 
                                           num_topics=n, 
                                           random_state=42, 
                                           id2word=dictionary, 
                                           passes=10, 
                                           alpha=0.01,
                                           eta=0.0001)
        lda_model.append(model)
        cm = gensim.models.ldamodel.CoherenceModel(model=model, dictionary=dictionary, coherence='c_v', texts=texts)
        coherence_list.append(cm.get_coherence())
        print('\nNumber of topic:', n)
        for idx, topic in model.print_topics(-1):
            print('\nTopic: {} \nWords: {}'.format(idx, topic)) 
    top_coherence_pos = coherence_list.index(max(coherence_list))
    ideal_topics = top_coherence_pos + 1
    lda_best_model = lda_model[top_coherence_pos]
    return coherence_list, lda_best_model, ideal_topics
        

def plot_coherence(num_topics, coherence):
    x_axis = range(1, num_topics)
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
    lda_display = pyLDAvis.gensim.prepare(model, corpus, dictionary, sort_topics=False)
    pyLDAvis.display(lda_display)

    
def run(df):
    preprocessed, bigram = preprocessing_reviews(df)
    bigram_reviews = make_bigrams(preprocessed, bigram)
    dictionary = create_dictionary(df, bigram_reviews)
    num_topics = 10
    
    # LDA using Bag of Words
    bow_corpus = bag_of_words(df, bigram_reviews)
    coherence_list, lda_best_model, ideal_topics = evaluate_multiple_lda(corpus=bow_corpus, 
                                                                         num_topics=num_topics, 
                                                                         dictionary=dictionary, 
                                                                         texts=bigram_reviews)
    plot_coherence(num_topics=num_topics, coherence=coherence_list)
    show_topics(lda_best_model, ideal_topics, num_words=15)
    topic_visualization(lda_best_model, bow_corpus, dictionary)
      
    '''
    # LDA using Tf-Idf
    corpus_tfidf = tf_idf(df, bow_corpus)
    coherence_tfidf = evaluate_topic(corpus=corpus_tfidf, num_topics=num_topics, dictionary=dictionary, texts=preprocessed_reviews)
    plot_coherence(num_topics=num_topics, coherence=coherence_tfidf)
    '''
'''
#%% FERRI E BASSO
def topic_based_tokenization(reviews):
    tokenizedReviews = {}
    key = 1
    #stopwords = nltk.corpus.stopwords.words("english")
    regexp = re.compile(r'\?')
    for review in reviews:
        for sentence in nltk.sent_tokenize(review):
            #logic to remove questions and errors
            if regexp.search(sentence):
                print("removed")
            else:
                sentence=re.sub(r'\(.*?\)','',sentence)
                tokenizedReviews[key]=sentence
                key += 1

    for key,value in tokenizedReviews.items():
        print(key,' ',value)
        tokenizedReviews[key]=value
    return tokenizedReviews

#%%
reviews_values = df_product.reviewText.values
reviews = [reviews_values[i] for i in range(len(reviews_values))]

#%%
reviews_filtered = removing_stop_words(reviews)
reviews_tokenized = topic_based_tokenization(reviews_filtered)
'''