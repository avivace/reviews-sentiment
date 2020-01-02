import data_exploration
import sentiment_analysis
import topic_sentiment_analysis
import pandas as pd
from data_utils import load_dataset
from data_utils import feature_manipulation
from data_utils import add_features
from pathlib import Path

def load_initial_dataset():
    dataset_folder = Path("../datasets/")
    try:
    	# Try to load a cached version of the dataframe
        print("Trying to load the cached dataframe...")
        df = pd.read_pickle(dataset_folder / 'cached_dataframe.pkl')
    except:
        print("No cached dataframe, loading the dataset from disk")
        path_file = dataset_folder / 'Cell_Phones_and_Accessories_5.json'
        df = load_dataset(path_file)
        # Store the dataframe on disk
        print("Caching the dataframe")
        df.to_pickle(dataset_folder / 'cached_dataframe.pkl')
    return df


def data_exploration_step(df):
    #Data exploration
    data_exploration.run(df)


def preprocessing_pre_exploration_dataset(df):
    preprocessed = df.copy(True)
    add_features(preprocessed)
    return preprocessed
    

def preprocessing_post_exploration_dataset(df):
    try:
        print("Trying to load the cached preprocessed dataframe")
        preprocessed = pd.read_pickle(dataset_folder / 'cached_preprocessed_dataframe.pkl')
    except:
        print("No cached dataframe, loading the dataset from disk")
        preprocessed = df.copy(True)
        feature_manipulation(preprocessed)
        print("Caching the preprocessed dataframe")
        preprocessed.to_pickle(dataset_folder / 'cached_preprocessed_dataframe.pkl')
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
    print(df)
    print(df_exploration)
    data_exploration_step(df_exploration)
    df_analysis = preprocessing_post_exploration_dataset(df_exploration)
    print(df_exploration)
    print(df_analysis)
    #sentiment_analysis_step(df_analysis)
    #aspect_based_sentiment_analysis_step(df_analysis)
