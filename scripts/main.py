import data_exploration
import sentiment_analysis
import topic_analysis
import pandas as pd
from data_utils import load_dataset
from data_utils import feature_manipulation
from data_utils import add_features
from pathlib import Path
import re

from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS

from bs4 import BeautifulSoup

app = Flask(__name__)
CORS(app)

def load_initial_dataset():
    dataset_folder = Path("../datasets/")
    try:
    	# Try to load a cached version of the dataframe
        print("Trying to load the cached dataframe...")
        df = pd.read_pickle(dataset_folder / 'cached_dataframe.pkl')
        print("Done")
    except:
        print("No cached dataframe, loading the dataset from disk")
        path_file = dataset_folder / 'Cell_Phones_and_Accessories.json'
        print(path_file)
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

@app.route('/')
def hello():
    result ={}
    result["positive"] = sentiment_analysis.compute_single(request.args.get('text'), best_nb, count_vector)[0][1]
    return jsonify(result)

def striphtml(reviews):
    n = 0
    filtered_reviews = []
    for text in df['reviewText']:
        m = re.search('<\s*a[^>]*>(.*?)<\s*/\s*a>', text)
        if m:
            soup = BeautifulSoup(text,features="html.parser")
            stripped_text = soup.get_text()
            filtered_reviews.append(stripped_text)
            n = n + 1
        else:
            filtered_reviews.append(text)
            
    print("HTML stripped on",n,"reviews")
    return filtered_reviews

def clean_dirt(df):
    reviews = df['reviewText'].tolist()
    htmlcleaned_reviews = striphtml(reviews)
    df['reviewText'] = [''.join(review) for review in htmlcleaned_reviews]
    

def check_dirt(df):
    c = 0
    for text in df['reviewText']:
        m = re.search('<\s*a[^>]*>(.*?)<\s*/\s*a>', text)
        if m:
            c = c+1        
    print(c,"dirty reviews")



if __name__ == "__main__":
    df = load_initial_dataset()
    #html_review_debug(df)
    # Remember to set this back to df
    df_exploration = preprocessing_pre_exploration_dataset(df)
    
    #data_exploration.run(df_exploration)
    df_analysis = preprocessing_post_exploration_dataset(df_exploration)
    
    check_dirt(df_analysis)
    clean_dirt(df_analysis)
    check_dirt(df_analysis)

    # Web server exposing the trained models
    #best_nb, best_lr, count_vector = sentiment_analysis.run(df_analysis)
    #app.run()

    # Enable Topic Analysis
    topic_analysis.run(df_analysis)
