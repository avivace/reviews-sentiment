import data_exploration
import sentiment_data_preparation
import sentiment_analysis
import topic_sentiment_data_preparation
import topic_sentiment_analysis
import os
from data_utils import load_dataset
import pandas

# Let's get back to the root folder of the project
os.chdir("..")
#os.chdir(r'./reviews-sentiment/')

# Data loading
try:
	# Try to load a cached version of the dataframe
	print("Trying to load the cached dataframe")
	df = pandas.read_pickle('cached_dataframe.pkl')
except:
	print("No cached dataframe, loading the dataset from disk")
	path = r'./datasets/Grocery_and_Gourmet_Food_5.json'
	df = load_dataset(path)
	# Store the dataframe on disk
	print("Caching the dataframe")
	df.to_pickle('cached_dataframe.pkl')

#%%
#Data exploration
data_exploration.run(df)

#Copy for Sentiment analysis
df_copy_for_sentiment_analysis = df.copy(deep=True)

#Data Preparation for Sentiment Analysis
sentiment_data_preparation.run(df_copy_for_sentiment_analysis)

#Sentiment Analysis
sentiment_analysis.run(df_copy_for_sentiment_analysis)

#Copy for Aspect based
df_copy_for_aspect_based = df.copy(deep=True)

#Data Preparation for Aspect based
topic_sentiment_data_preparation.run(df_copy_for_aspect_based)

#Aspect based
topic_sentiment_analysis.run(df_copy_for_aspect_based)
