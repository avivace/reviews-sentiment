import data_exploration
import sentiment_analysis
import topic_sentiment_analysis
import os
import pandas as pd
from data_utils import load_dataset
from data_utils import feature_manipulation



def load_initial_dataset():
    try:
    	# Try to load a cached version of the dataframe
        print("Trying to load the cached dataframe")
        df = pd.read_pickle('cached_dataframe.pkl')
    except:
        print("No cached dataframe, loading the dataset from disk")
        path_file = r'./datasets/Grocery_and_Gourmet_Food_5.json'
        #path = r'./datasets/Cell_Phones_and_Accessories_5.json'
        #path_file = r'./datasets/Magazine_Subscriptions.json'
        df = load_dataset(path_file)
        # Store the dataframe on disk
        print("Caching the dataframe")
        df.to_pickle('cached_dataframe.pkl')
    return df


def data_exploration_step(df):
    #Data exploration
    data_exploration.run(df)
    
    
def preprocessing_dataset(df):
    try:
        print("Trying to load the cached preprocessed dataframe")
        preprocessed = pd.read_pickle('cached_preprocessed_dataframe.pkl')
    except:
        print("No cached dataframe, loading the dataset from disk")
        #preprocessed = pd.DataFrame()
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
    os.chdir("..")
    df = load_initial_dataset()
    df_preprocessed = preprocessing_dataset(df)
    os.chdir(r'./scripts')
    data_exploration_step(df_preprocessed)

    sentiment_analysis_step(df_preprocessed)
    aspect_based_sentiment_analysis_step(df_preprocessed)
