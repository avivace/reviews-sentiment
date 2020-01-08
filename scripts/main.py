import data_exploration
import sentiment_analysis
import topic_analysis
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
        print("Done")
    except:
        print("No cached dataframe, loading the dataset from disk")
        path_file = dataset_folder / 'Cell_Phones_and_Accessories_5.json'
        df = load_dataset(path_file)
        # Store the dataframe on disk
        print("Caching the dataframe")
        df.to_pickle(dataset_folder / 'cached_dataframe.pkl')
    return df


def preprocessing_pre_exploration_dataset(df):
    preprocessed = df.copy(True)
    add_features(preprocessed)
    return preprocessed
    

def preprocessing_post_exploration_dataset(df):
    dataset_folder = Path("../datasets/")
    try:
        print("Trying to load the cached preprocessed dataframe...")
        preprocessed = pd.read_pickle(dataset_folder / 'cached_preprocessed_dataframe.pkl')
        print("Done")
    except:
        print("No cached dataframe, loading the dataset from disk")
        preprocessed = df.copy(True)
        feature_manipulation(preprocessed)
        print("Caching the preprocessed dataframe")
        preprocessed.to_pickle(dataset_folder / 'cached_preprocessed_dataframe.pkl')
    return preprocessed


if __name__ == "__main__":
    df = load_initial_dataset()
    df_exploration = preprocessing_pre_exploration_dataset(df)
    data_exploration.run(df_exploration)
    df_analysis = preprocessing_post_exploration_dataset(df_exploration)
    sentiment_analysis.run(df_analysis)
    topic_analysis.run(df_analysis)
