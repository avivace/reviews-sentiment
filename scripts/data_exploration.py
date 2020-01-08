# -*- coding: utf-8 -*-

### Import libraries ###

import numpy as np
import pandas as pd
from pandas import Grouper
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style(style="darkgrid")
from data_utils import most_reviewed_products
from pathlib import Path
from matplotlib import rcParams

figOutputPath = Path("../figures/")

### Functions ###

def label_typography(ax):
    ax.xaxis.label.set_fontweight(500)
    ax.yaxis.label.set_fontsize(15)
    ax.yaxis.label.set_fontweight(500)
    ax.xaxis.label.set_fontsize(15)
    return


def most_active_reviewers(df, n_reviewers):
    n_reviews = df['reviewerID'].value_counts()
    most_reviews = n_reviews.nlargest(n_reviewers)
    most_reviews = most_reviews.reset_index()
    most_reviews = most_reviews.drop('reviewerID', axis=1)
    definitive = df.merge(most_reviews, left_on='reviewerID', right_on='index')
    definitive = definitive.drop('index', axis=1)
    return definitive


def analyze_reviews(df, df_attribute, name_file, ylabel):
    print("Shape of df: ", df.shape)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.countplot(df_attribute, ax=ax)
    label_typography(ax)

    # Set and style the title, and move it up a bit (1.02 = 2%)
    #ax.set_title(title, fontname='Inter', fontsize=20, fontweight=500, y=1.02)
    
    ax.xaxis.label.set_text(ylabel)
    ax.yaxis.label.set_text("Review Count")
    if (name_file=="review_distribution_per_day"):
        ax.set_xticklabels(["Sunday", "Monday", "Thuesday", "Wednesday", "Thursday", "Friday", "Saturday"])
        ax.set_yticks([0, 100000, 200000])
        ax.set_yticklabels(["0", "100K", "200K"])
    else:
        ax.set_yticks([0, 100000, 500000, 1000000])
        ax.set_yticklabels(["0", "100K", "500K", "1M"])



    ax.figure.savefig(figOutputPath / '1_{0}.svg'.format(name_file), format='svg')
    print('Exported 1_{}.svg'.format(name_file))

    print("Average Score: ", np.mean(df.overall))
    print("Median Score: ", np.median(df.overall))


def run(df):
    # 1 - Countplot: score distribution    
    analyze_reviews(df, df.overall, 'score_distribution', 'Score')
    
    # 2 - Countplot: opinion distribution    
    analyze_reviews(df, df.opinion, 'opinion_distribution', 'Opinion')

    # 3 - Distribution of words
    reduced_df = df.copy()
    reduced_df = reduced_df[reduced_df['n_words'] <= 1000]
    fig, ax5 = plt.subplots()
    ax5 = sns.violinplot(x=reduced_df['opinion'], y=reduced_df['n_words'])
    #ax5.set_title('Distribution of words in review for each opinion')
    ax5.figure.savefig(figOutputPath / '1_correlation_words_opinion.svg', format='svg')
    
    # 4 - Review distribution per day
    analyze_reviews(df, df.week_day, 'review_distribution_per_day', 'Day')
    
    # 5 - Top 20 products
    fig, ax3 = plt.subplots(figsize=(15, 15))
    top_products = most_reviewed_products(df, 20)
    r = list(top_products['asin'].unique())
    positive = list(top_products.loc[top_products['opinion'] == 'positive', 'asin'].value_counts().reindex(top_products['asin'].unique(), fill_value=0))
    neutral = list(top_products.loc[top_products['opinion'] == 'neutral', 'asin'].value_counts().reindex(top_products['asin'].unique(), fill_value=0))
    negative = list(top_products.loc[top_products['opinion'] == 'negative', 'asin'].value_counts().reindex(top_products['asin'].unique(), fill_value=0))
    raw_data = {'positive': positive, 'neutral': neutral, 'negative': negative}
    raw_data = pd.DataFrame(raw_data)
    
    totals = [i+j+k for i,j,k in zip(raw_data['positive'], raw_data['neutral'], raw_data['negative'])]
    positive_percentage = [i / j * 100 for i, j in zip(raw_data['positive'], totals)]
    neutral_percentage = [i / j * 100 for i, j in zip(raw_data['neutral'], totals)]
    negative_percentage = [i / j * 100 for i, j in zip(raw_data['negative'], totals)]

    bar_width = 0.85

    ax3.bar(r, positive_percentage, color='#b5ffb9', edgecolor='white', width=bar_width, label='positive')
    ax3.bar(r, neutral_percentage, bottom=positive_percentage, color='#f9bc86', edgecolor='white', width=bar_width, label='neutral')
    ax3.bar(r, negative_percentage, bottom=[i + j for i, j in zip(positive_percentage, neutral_percentage)], color='#a3acff', edgecolor='white', width=bar_width, label='negative')
    ax3.set_xticklabels(r, rotation=90)
    ax3.set_xlabel('Code product')
    legend = ax3.legend(loc='lower left', shadow=True, fontsize='large')
    legend.get_frame().set_facecolor('#00FFCC')
    #ax3.set_title('Opinion for besteller products')
    ax3.figure.savefig(figOutputPath / '1_sentiment_reviews_bestseller_products.svg', format='svg')
    print("Exported 1_sentiment_reviews_besteller_products.svg")

    # 6 - Top 50 reviewers
    fig, ax4 = plt.subplots(figsize=(15, 15))
    top_reviewers = most_active_reviewers(df, 50)
    sns.countplot(top_reviewers.reviewerID, ax=ax4, order=top_reviewers['reviewerID'].value_counts().index)
    r = list(top_reviewers['reviewerID'].unique())
    ax4.set_xticklabels(r, rotation=90)
    #ax4.set_title('Reviewers with most reviews')
    ax4.figure.savefig(figOutputPath / '1_reviewers_most_reviews.svg', format='svg')
    
    # 7 - Opinion of top reviewers
    fig, ax6 = plt.subplots(figsize=(15, 15))
    top_reviewers = most_active_reviewers(df, 50)
    r = list(top_reviewers['reviewerID'].unique())
    positive = list(top_reviewers.loc[top_reviewers['opinion'] == 'positive', 'reviewerID'].value_counts().reindex(top_reviewers['reviewerID'].unique(), fill_value=0))
    neutral = list(top_reviewers.loc[top_reviewers['opinion'] == 'neutral', 'reviewerID'].value_counts().reindex(top_reviewers['reviewerID'].unique(), fill_value=0))
    negative = list(top_reviewers.loc[top_reviewers['opinion'] == 'negative', 'reviewerID'].value_counts().reindex(top_reviewers['reviewerID'].unique(), fill_value=0))
    raw_data = {'positive': positive, 'neutral': neutral, 'negative': negative}
    raw_data = pd.DataFrame(raw_data)

    #print("Opinions ",raw_data)
    
    totals = [i+j+k for i,j,k in zip(raw_data['positive'], raw_data['neutral'], raw_data['negative'])]
    #totals = list(top_products['asin'].value_counts().reindex(top_products['asin'].unique(), fill_value=0))
    positive_percentage = [i / j * 100 for i, j in zip(raw_data['positive'], totals)]
    neutral_percentage = [i / j * 100 for i, j in zip(raw_data['neutral'], totals)]
    negative_percentage = [i / j * 100 for i, j in zip(raw_data['negative'], totals)]

    bar_width = 0.85

    ax6.bar(r, positive_percentage, color='#b5ffb9', edgecolor='white', width=bar_width, label='positive')
    ax6.bar(r, neutral_percentage, bottom=positive_percentage, color='#f9bc86', edgecolor='white', width=bar_width, label='neutral')
    ax6.bar(r, negative_percentage, bottom=[i + j for i, j in zip(positive_percentage, neutral_percentage)], color='#a3acff', edgecolor='white', width=bar_width, label='negative')
    ax6.set_xticklabels(r, rotation=90)
    ax6.set_xlabel('Reviewer ID')
    legend = ax6.legend(loc='lower left', shadow=True, fontsize='large')
    legend.get_frame().set_facecolor('#00FFCC')
    #ax6.set_title('Opinion of top reviewers')
    plt.show()
    ax6.figure.savefig(figOutputPath / '1_opinion_top_reviewers.svg', format='svg')
    print("Exported 1_opinion_top_reviewers.svg")
    
    # 8 - Unverified reviews
    unverified = df[df['verified'] == False]
    analyze_reviews(unverified, df.overall, 'unverified_score_distribution', 'Score')

    # 9 - Verified reviews
    verified = df[df['verified'] == True]
    analyze_reviews(verified, df.overall, 'verified_score_distribution', 'Score')

    # 10 - verified vs unverified of top 50 reviewers
    fig, ax7 = plt.subplots(figsize=(15, 15))
    r = list(top_reviewers['reviewerID'].unique())
    verified = list(top_reviewers.loc[top_reviewers['verified'] == True, 'reviewerID'].value_counts().reindex(top_reviewers['reviewerID'].unique(), fill_value=0))
    unverified = list(top_reviewers.loc[top_reviewers['verified'] == False, 'reviewerID'].value_counts().reindex(top_reviewers['reviewerID'].unique(), fill_value=0))
    raw_data = {'verified': verified, 'unverified': unverified}
    raw_data = pd.DataFrame(raw_data)

    totals = [i+j for i,j in zip(raw_data['verified'], raw_data['unverified'])]
    verified_percentage = [i / j * 100 for i, j in zip(raw_data['verified'], totals)]
    unverified_percentage = [i / j * 100 for i, j in zip(raw_data['unverified'], totals)]

    bar_width = 0.85

    ax7.bar(r, verified_percentage, color='#b5ffb9', edgecolor='white', width=bar_width, label='verified')
    ax7.bar(r, unverified_percentage, bottom=verified_percentage, color='#f9bc86', edgecolor='white', width=bar_width, label='unverified')
    ax7.set_xticklabels(r, rotation=90)
    ax7.set_xlabel('Reviewer ID')
    legend = ax7.legend(loc='upper right', shadow=True, fontsize='large')
    legend.get_frame().set_facecolor('#00FFCC')
    #ax7.set_title('Verified vs Unverified reviews of top reviewers')
    plt.show()
    ax7.figure.savefig(figOutputPath / '1_verified_unverified.svg', format='svg')
    print("Exported 1_verified_unverified.svg")
    