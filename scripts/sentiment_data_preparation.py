# -*- coding: utf-8 -*-

from data_utils import remove_cols
from data_utils import wordcloud
from data_utils import vote_to_opinion
from data_utils import preprocessing
from data_utils import lemmatization
import nltk
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#%%
def undersampling(df):
    df = remove_cols(df)
    #Create new feature "opinion" based on vote
    df = vote_to_opinion(df)
    df.drop(df[df.opinion == 'neutral'].index, inplace=True)
    positive, negative = df.opinion.value_counts()
    df_positive = df[df.opinion == 'positive']
    df_positive = df_positive.sample(negative, random_state=42)
    df_negative = df[df.opinion == 'negative']
    df = pd.concat([df_positive, df_negative])
    df = df.sample(frac=1)
    return df


def data_preparation(df):
    df = undersampling(df)
    #alternative: df[df['opinion'].map(lambda x: str(x)!="neutral")]
    reviews = df['reviewText'].tolist()
    preprocessed = preprocessing(reviews)
    df['preprocessedReview'] = [' '.join(review) for review in lemmatized]
    return df


def retrieve_opinion(df, sentiment):
    opinion = df[df['opinion'] == sentiment]
    reviews = opinion['preprocessedReview'].tolist()
    wordcloud(reviews)
    

def vectorization(df):
    cvector = CountVectorizer(max_features=10000, min_df=7, max_df=0.8)
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
    

def run(df):
    # Preprocessing

    
    '''
    # Create sentiment for each review.
    # Quattro diverse colonne neutral, pos, neg e compound.
    # Valore tra 0 e 1 che stima la probabilità del sentiment, il compound è calcolato rispetto a questi tre valori
    # Testata su review intera, non pre-processata. Vedere se funziona e come

    
    ### Predizione su usefulness e sentiment (Positive > 3, Usefulness?? vedere quantile, 90% sono useless)
    print(df.vote.quantile(0.95))
    
    analyser = SentimentIntensityAnalyzer()
    df['sentiments'] = df['preprocessedReview'].apply(lambda x: analyser.polarity_scores(x))
    df = pd.concat([df.drop(['sentiments'], axis=1), df['sentiments'].apply(pd.Series)], axis=1)
    # Number of characters
    df["nb_chars"] = df["preprocessedReview"].apply(lambda x: len(x))
    # Numbers of words
    df["nb_words"] = df["preprocessedReview"].apply(lambda x: len(x.split(" ")))
    # Most positive reviews
    df[df["nb_words"] >= 5].sort_values("pos", ascending = False)[["reviewText", "pos"]].head(10)
    # Most negative reviews
    df[df["nb_words"] >= 5].sort_values("neg", ascending = True)[["reviewText", "neg"]].head(10)
    '''
    


    
'''
    #Remove stop words
    df['reviewTextNoStopWords'] = remove_stop_words(df['reviewText'])

    #Sentence tokenization
    df['reviewTextTokenized'] = tokenization(df['reviewTextNoStopWords'])


    #Drop non-final review texts
    df.drop(['reviewText', 'reviewTextNoStopWords'], axis=1)
    
    #Save prepared dataset for Sentiment Analysis
    #df.to_csv("datasets\preparedForSentimentAnalysis.csv", sep='\t', encoding='utf-8')

OLD #%%
#Get the reviews
reviews_values = df.reviewText.values
reviews = [reviews_values[i] for i in range(len(reviews_values))]
#%%
#Remove stop words
filtered_reviews = removing_stop_words(reviews)

#%%
#Sentence tokenization
reviews = tokenization(filtered_reviews)
#%%'''

''' OLD reviews = df.reviewText.values
labels = df.opinion.values

if df.opinion[3] == "positive":
    print("\nPositive:", reviews[3][:90], "...")
else:
    print("\nNegative:", reviews[3][:90], "...")
    
positive_reviews = [reviews[i] for i in range(len(reviews)) if labels[i] == "positive"]
negative_reviews = [reviews[i] for i in range(len(reviews)) if labels[i] == "negative"]

#%% Tokenization

positive_tokenized = tokenization(positive_reviews)
negative_tokenized = tokenization(negative_reviews)

#%% Stop words

positive_filtered = removing_stop_word(positive_tokenized)
negative_filtered = removing_stop_words(negative_tokenized)'''

