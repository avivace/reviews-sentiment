# -*- coding: utf-8 -*-

### Import libraries ###
import nltk
import gensim
nltk.download('wordnet')

from topic_sentiment_data_preparation import bag_of_words, create_dictionary, tf_idf, preprocessing_reviews

### Functions ###

def run(df):
    preprocessed_reviews = preprocessing_reviews(df)
    dictionary = create_dictionary(df, preprocessed_reviews)
    # LDA using Bag of Words
    bow_corpus = bag_of_words(df, preprocessed_reviews)
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))
    
    corpus_tfidf = tf_idf(df, bow_corpus)
    # LDA using Tf-Idf
    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)
    for idx, topic in lda_model_tfidf.print_topics(-1):
        print('Topic: {} Word: {}'.format(idx, topic))

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