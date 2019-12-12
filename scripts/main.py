import data_exploration
import sentiment_analysis
import topic_sentiment_analysis
import os
import pandas as pd
from data_utils import load_dataset


# Let's get back to the root folder of the project
os.chdir("..")
#os.chdir(r'./reviews-sentiment/')

# Data loading
try:
	# Try to load a cached version of the dataframe
    print("Trying to load the cached dataframe")
    df = pd.read_pickle('cached_dataframe.pkl')
except:
    print("No cached dataframe, loading the dataset from disk")
    path = r'./datasets/Grocery_and_Gourmet_Food_5.json'
    #path = r'./datasets/Cell_Phones_and_Accessories_5.json'
    df = load_dataset(path)
    # Store the dataframe on disk
    print("Caching the dataframe")
    df.to_pickle('cached_dataframe.pkl')
    

#%%
def data_exploration_step():
    #Data exploration
    data_exploration.run(df)

def sentiment_analysis_step():
    #Copy for Sentiment analysis
    df_copy_for_sentiment_analysis = df.copy(deep=True)
    #Sentiment Analysis
    sentiment_analysis.run(df_copy_for_sentiment_analysis)

def aspect_based_sentiment_analysis_step():
    #Copy for Aspect based
    df_copy_for_aspect_based = df.copy(deep=True)
    #Aspect based
    topic_sentiment_analysis.run(df_copy_for_aspect_based)

if __name__ == "__main__":
    os.chdir(r'./scripts')
    #data_exploration_step()
    #sentiment_analysis_step()
    aspect_based_sentiment_analysis_step()
