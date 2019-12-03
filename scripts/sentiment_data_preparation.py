# -*- coding: utf-8 -*-

from data_utils import remove_cols, vote_to_opinion, preprocessing
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#%%
def wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color = 'white',
        max_words = 200,
        max_font_size = 40, 
        scale = 3,
        random_state = 42
    ).generate(str(data))

    fig = plt.figure(1, figsize = (20, 20))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize = 20)
        fig.subplots_adjust(top = 2.3)

    plt.imshow(wordcloud)
    plt.show()
    

def run(df):
    #Features selection
    df = remove_cols(df)
    #Create new feature "opinion" based on vote
    df = vote_to_opinion(df)
    # Wordcloud before remove neutral reviews
    # Remove neutral reviews
    df.drop(df[df.opinion == 'neutral'].index, inplace=True)
    #alternative: df[df['opinion'].map(lambda x: str(x)!="neutral")]
    reviews = df['reviewText'].tolist()
    # Preprocessing
    df['preprocessedReview'] = preprocessing(reviews)
    #alternative: df['preprocessed_review'] = df['reviewText'].apply(lambda x: preprocessing(x))
    # Create sentiment for each review.
    # Quattro diverse colonne neutral, pos, neg e compound.
    # Valore tra 0 e 1 che stima la probabilità del sentiment, il compound è calcolato rispetto a questi tre valori
    # Testata su review intera, non pre-processata. Vedere se funziona e come
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

