# -*- coding: utf-8 -*-

### Import libraries ###

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style("dark")

from data_utils import vote_to_opinion

### Functions ###

def most_reviewed_products(df, n_products):
    reviews_per_product = df['asin'].value_counts()
    most_reviews = reviews_per_product.nlargest(n_products)
    most_reviews = most_reviews.reset_index()
    most_reviews = most_reviews.drop('asin', axis=1)
    
    definitive = df.merge(most_reviews, left_on='asin', right_on='index')
    definitive = definitive.drop('index', axis=1)
    
    return definitive


def significative_reviews(df, n_votes):
    return df[df['vote'] >= n_votes]


def most_active_reviewers(df, n_reviewers):
    n_reviews = df['reviewerID'].value_counts()
    most_reviews = n_reviews.nlargest(n_reviewers)
    most_reviews = most_reviews.reset_index()
    most_reviews = most_reviews.drop('reviewerID', axis=1)
    #print("CCC \n", most_reviews)
    
    definitive = df.merge(most_reviews, left_on='reviewerID', right_on='index')
    definitive = definitive.drop('index', axis=1)
    return definitive


def run(df):
    #Create new feature "opinion" based on vote
    df = vote_to_opinion(df)

    print("DATA EXPLORATION")

    print("Number of records:", len(df))

    # Score Distribution
    ax = plt.axes()
    sns.countplot(df.overall, ax=ax)
    ax.set_title('Score Distribution')
    # plt.show()
    plt.savefig('figures/1_scoredistribution.svg', format='svg')
    print("Exported 1_scoredistribution.svg")

    print("Average Score: ", np.mean(df.overall))
    print("Median Score: ", np.median(df.overall))

    #Opinion Distribution
    ax = plt.axes(label="a")
    sns.countplot(df.opinion, ax=ax)
    ax.set_title('Opinion Distribution')
    #plt.show()
    plt.savefig('figures/1_opiniondistribution.svg', format='svg')
    print("Exported 1_opiniondistribution.svg")

    print("Proportion of positive review:", len(df[df.opinion == "positive"]) / len(df))
    print("Proportion of neutral review:", len(df[df.opinion == "neutral"]) / len(df))
    print("Proportion of negative review:", len(df[df.opinion == "negative"]) / len(df))

    #Stacked barplot (x-axis asin code, y-axis opinion)

    top_products = most_reviewed_products(df, 20)
    r = list(top_products['asin'].unique())
    positive = list(top_products.loc[top_products['opinion'] == 'positive', 'asin'].value_counts().reindex(top_products['asin'].unique(), fill_value=0))
    neutral = list(top_products.loc[top_products['opinion'] == 'neutral', 'asin'].value_counts().reindex(top_products['asin'].unique(), fill_value=0))
    negative = list(top_products.loc[top_products['opinion'] == 'negative', 'asin'].value_counts().reindex(top_products['asin'].unique(), fill_value=0))
    raw_data = {'positive': positive, 'neutral': neutral, 'negative': negative}

    #print("Opinions ",raw_data)

    totals = list(top_products['asin'].value_counts().reindex(top_products['asin'].unique(), fill_value=0))
    positive_percentage = [i / j * 100 for i, j in zip(positive, totals)]
    neutral_percentage = [i / j * 100 for i, j in zip(neutral, totals)]
    negative_percentage = [i / j * 100 for i, j in zip(negative, totals)]

    bar_width = 0.85
    names = tuple(r)

    plt.bar(r, positive_percentage, color='#b5ffb9', edgecolor='white', width=bar_width)
    plt.bar(r, neutral_percentage, bottom=positive_percentage, color='#f9bc86', edgecolor='white', width=bar_width)
    plt.bar(r, negative_percentage, bottom=[i + j for i, j in zip(positive_percentage, neutral_percentage)], color='#a3acff', edgecolor='white', width=bar_width)
    plt.xticks(r, names, rotation=90)
    plt.xlabel('code product')
    # plt.show()
    plt.savefig('figures/1_reviews.svg', format='svg')
    print("Exported 1_reviews.svg")

    top_reviewers = most_active_reviewers(df, 50)
    #print("DDD \n",top_reviewers)
    top_reviewers = top_reviewers.groupby('reviewerID').agg({'overall':'mean',
                                                             'vote':'mean',
                                                             'asin':'count'}).sort_values(['asin'], ascending=[False]).reset_index()
    #print("EEE \n",top_reviewers)

    r = list(top_reviewers['reviewerID'].unique())
    bar_width = 0.85
    names = tuple(r)

    plt.bar(r, top_reviewers['asin'], color='#f9bc86', edgecolor='white', width=bar_width)
    plt.bar(r, top_reviewers['overall'], color='#b5ffb9', edgecolor='white', width=bar_width)
    plt.xticks(r, names, rotation=90)
    plt.xlabel('Reviewer ID')
    # plt.show()
    plt.savefig('figures/1_reviewers.svg', format='svg')
    print("Exported 1_reviewers.svg")
    
    '''# Correlation between votes and opinion with a boxplot.
    # Maybe a better representation than boxplot can be considered
    # This part of code DOESN'T work

    df_significative_reviews = significative_reviews(df, 5)
    plt.boxplot(x=df_significative_reviews['opinion'], 
                y=pd.to_numeric(df_significative_reviews['vote'],
                errors='coerce'), 
                palette="Blues")
    plt.show()'''
