import data_exploration
import sentiment_analysis
import topic_sentiment_analysis
import os
import pandas as pd
from data_utils import load_dataset
from data_utils import feature_manipulation
from data_utils import add_features


def load_initial_dataset():
    os.chdir("..")
    try:
    	# Try to load a cached version of the dataframe
        print("Trying to load the cached dataframe")
        df = pd.read_pickle('cached_dataframe.pkl')
    except:
        print("No cached dataframe, loading the dataset from disk")
        #path_file = r'./datasets/Grocery_and_Gourmet_Food_5.json'
        path_file = r'./datasets/Cell_Phones_and_Accessories_5.json'
        #path_file = r'./datasets/Magazine_Subscriptions.json'
        df = load_dataset(path_file)
        # Store the dataframe on disk
        print("Caching the dataframe")
        df.to_pickle('cached_dataframe.pkl')
    return df


def data_exploration_step(df):
    #Data exploration
    data_exploration.run(df)


def preprocessing_pre_exploration_dataset(df):
    df_add_features = add_features(df)
    preprocessed = pd.DataFrame()
    preprocessed = pd.concat([preprocessed, df_add_features], axis=1)
    return preprocessed
    

def preprocessing_post_exploration_dataset(df):
    os.chdir("..")
    try:
        print("Trying to load the cached preprocessed dataframe")
        preprocessed = pd.read_pickle('cached_preprocessed_dataframe.pkl')
    except:
        print("No cached dataframe, loading the dataset from disk")
        preprocessed = feature_manipulation(df)
        print("Caching the dataframe")
        preprocessed.to_pickle('cached_preprocessed_dataframe.pkl')
    return preprocessed


def sentiment_analysis_step(df_preprocessed):
    #Copy for Sentiment analysis
    #df_copy_for_sentiment_analysis = df.copy(deep=True)
    #Sentiment Analysis
    sentiment_analysis.run(df_preprocessed)

def aspect_based_sentiment_analysis_step(df_preprocessed):
    #Copy for Aspect based
    #df_copy_for_aspect_based = df.copy(deep=True)
    #Aspect based
    #topic_sentiment_analysis.run(df_copy_for_aspect_based)
    topic_sentiment_analysis.run(df_preprocessed)

if __name__ == "__main__":
    df = load_initial_dataset()
    df_exploration = preprocessing_pre_exploration_dataset(df)
    data_exploration_step(df_exploration)
    df_analysis = preprocessing_post_exploration_dataset(df_exploration)
    os.chdir(r'./scripts')
    #sentiment_analysis_step(df_analysis)
    aspect_based_sentiment_analysis_step(df_analysis)
