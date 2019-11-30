import data_exploration
import os
from data_utils import load_dataset
import pandas

# Let's get back to the root folder of the project
os.chdir("..")

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

# Data exploration
data_exploration.run(df)