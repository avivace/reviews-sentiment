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
import json

# Default text styling for figures
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Inter']
rcParams['font.weight'] = 500
rcParams['xtick.labelsize'] = 13
rcParams['ytick.labelsize'] = 13

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

def analyze_reviews(df, df_attribute, name_file, xlabel):
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.countplot(df_attribute, ax=ax)
    label_typography(ax)

    # Set and style the title, and move it up a bit (1.02 = 2%)
    #ax.set_title(title, fontname='Inter', fontsize=20, fontweight=500, y=1.02)
    
    ax.xaxis.label.set_text(xlabel)
    ax.yaxis.label.set_text("Review count")
    if (name_file=="review_distribution_per_day"):
        ax.set_xticklabels(["Sunday", "Monday", "Thuesday", "Wednesday", "Thursday", "Friday", "Saturday"])
        ax.xaxis.label.set_fontsize(13)
        ax.set_yticks([0, 100000, 200000])
        ax.set_yticklabels(["0", "100K", "200K"])
    elif (name_file=="review_distribution_per_month"):
        ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        ax.xaxis.label.set_fontsize(13)
        ax.set_yticks([0, 100000, 200000])
        ax.set_yticklabels(["0", "100K", "200K"])
    elif (name_file=="review_distribution_per_year"):
        ax.set_xticklabels([2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018])
        ax.xaxis.label.set_fontsize(13)
        ax.set_yticks([0, 100000, 200000])
        ax.set_yticklabels(["0", "100K", "200K"])
    elif (name_file=="unverified_overall_distribution"):
        ax.set_yticks([0, 50000, 100000])
        ax.set_yticklabels(["0", "50K", "100K"])
    elif (name_file=="verified_overall_distribution"):
        ax.set_yticks([0, 300000, 600000])
        ax.set_yticklabels(["0", "300K", "600K"])
    else:
        ax.set_yticks([0, 100000, 500000, 1000000])
        ax.set_yticklabels(["0", "100K", "500K", "1M"])



    ax.figure.savefig(figOutputPath / '1_{0}.svg'.format(name_file), format='svg')
    print('Exported 1_{}.svg'.format(name_file))

def run(df):
    # 1 - Countplot: overall distribution    
    analyze_reviews(df, df.overall, 'overall_distribution', 'Overall')
    
    # 2 - Countplot: opinion distribution    
    analyze_reviews(df, df.opinion, 'opinion_distribution', 'Opinion')

    # 3 - Distribution of words
    reduced_df = df.copy()
    reduced_df = reduced_df[reduced_df['n_words'] <= 1000]
    fig, ax5 = plt.subplots()
    ax5 = sns.violinplot(x=reduced_df['opinion'], y=reduced_df['n_words'])
    #ax5.set_title('Distribution of words in review for each opinion')
    ax5.xaxis.label.set_text("Opinion")
    ax5.yaxis.label.set_text("Number of words")
    label_typography(ax5)
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
    ax3.set_xlabel('Unique product')
    ax3.set_xticks([])
    ax3.set_ylabel('Percentage')
    ax3.set_xticks([])
    label_typography(ax3)
    #legend = ax3.legend(loc='lower left', shadow=True, fontsize='large')
    #legend.get_frame().set_facecolor('#00FFCC')
    #ax3.set_title('Opinion for besteller products')
    ax3.figure.savefig(figOutputPath / '1_sentiment_reviews_bestseller_products.svg', format='svg')
    print("Exported 1_sentiment_reviews_besteller_products.svg")

    # 6 - Top 50 reviewers
    fig, ax4 = plt.subplots(figsize=(15, 15))
    top_reviewers = most_active_reviewers(df, 50)
    sns.countplot(top_reviewers.reviewerID, ax=ax4, order=top_reviewers['reviewerID'].value_counts().index)
    r = list(top_reviewers['reviewerID'].unique())
    ax4.set_xticklabels(r, rotation=90)
    ax4.set_ylabel('Review count')
    ax4.set_xlabel('Unique Reviewers')
    ax4.set_xticks([])
    label_typography(ax4)
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

    bar_width = 1

    ax6.bar(r, positive_percentage, color='#b5ffb9', edgecolor='white', width=bar_width, label='positive')
    ax6.bar(r, neutral_percentage, bottom=positive_percentage, color='#f9bc86', edgecolor='white', width=bar_width, label='neutral')
    ax6.bar(r, negative_percentage, bottom=[i + j for i, j in zip(positive_percentage, neutral_percentage)], color='#a3acff', edgecolor='white', width=bar_width, label='negative')
    ax6.set_xticklabels(r, rotation=90)
    ax6.set_xlabel('Unique Reviewers')
    ax3.set_xticks([])
    ax6.set_xticks([])
    ax6.set_ylabel('Percentage')
    label_typography(ax6)
    label_typography(ax3)
    #legend = ax6.legend(loc='lower left', shadow=True, fontsize='large')
    #legend.get_frame().set_facecolor('#00FFCC')
    #ax6.set_title('Opinion of top reviewers')
    #plt.show()
    ax6.figure.savefig(figOutputPath / '1_opinion_top_reviewers.svg', format='svg')
    print("Exported 1_opinion_top_reviewers.svg")
    
    # 8 - Unverified reviews
    unverified = df[df['verified'] == False]
    analyze_reviews(unverified, unverified.overall, 'unverified_overall_distribution', 'Overall')

    # 9 - Verified reviews
    verified = df[df['verified'] == True]
    analyze_reviews(verified, verified.overall, 'verified_overall_distribution', 'Overall')

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

    bar_width = 1

    ax7.bar(r, verified_percentage, color='#b5ffb9', edgecolor='white', width=bar_width, label='verified')
    ax7.bar(r, unverified_percentage, bottom=verified_percentage, color='#f9bc86', edgecolor='white', width=bar_width, label='unverified')
    ax7.set_xticklabels(r, rotation=90)

    ax7.set_xlabel('Unique Reviewers')
    ax7.set_xticks([])
    ax3.set_xticks([])
    ax7.set_ylabel('Percentage')
    label_typography(ax3)
    label_typography(ax7)
    #legend = ax7.legend(loc='upper right', shadow=True, fontsize='large')
    #legend.get_frame().set_facecolor('#00FFCC')
    #ax7.set_title('Verified vs Unverified reviews of top reviewers')
    #plt.show()
    ax7.figure.savefig(figOutputPath / '1_verified_unverified.svg', format='svg')
    print("Exported 1_verified_unverified.svg")
    

# Exporting raw data for the web demo

def top_50_products_verified_unverified_both(df):
    print("top_50_products_verified_unverified_both")
    top_products = most_reviewed_products(df, 5)
    r = list(top_products['asin'].unique())
    products = []
    verified_series = []
    unverified_series = []
    overall_series = []

    for asin in r:
        print("Product: ", asin)
        products.append(asin)
        verified = df.loc[(df['asin'] == asin) & (df['verified'] == True), 'overall'].mean()
        print("-verified: ",verified)
        verified_series.append(verified)
        unverified = df.loc[(df['asin'] == asin) & (df['verified'] == False), 'overall'].mean()
        unverified_series.append(unverified)
        print("-unverified: ", unverified)
        aall = df.loc[(df['asin'] == asin), 'overall'].mean()
        overall_series.append(aall)
        print("-all: ", aall)

    obj = [
        {"name": "products",
        "data": products},
        {"name": "verified",
        "data": verified_series},
        {"name": "unverified",
        "data": unverified_series},
        {"name": "all",
        "data": overall_series
    }]

    with open('ver_unver.json', 'w') as outfile:
        json.dump(obj, outfile, indent=2, sort_keys=True)
    
    print(products)

def count_reviews(df):
    top_products = most_reviewed_products(df, 20)
    r = list(top_products['asin'].unique())
    products = []
    # One element per product
    verified_score_qty = []
    unverified_score_qty = []
    n = 0

    for asin in r:
        print("Product: ", asin)
        products.append(asin)
        dataseries_ver = []
        dataseries_unver = []

        for i in range(1,6):
            key = { "name" : int(i), "data": [int(df.loc[(df['asin'] == asin) & (df['verified'] == True) & (df['overall'] == i), 'overall'].count()), int(df.loc[(df['asin'] == asin) & (df['verified'] == False) & (df['overall'] == i), 'overall'].count())]}
            dataseries_ver.append(key)

        verified_score_qty.append(dataseries_ver)
        n = n+1

    obj = {'products': products, 'count':verified_score_qty,}
    
    with open('ver_counts.json', 'w') as outfile:
        json.dump(obj, outfile, indent=2, sort_keys=True)


def year_month_day_reviews(df):
    analyze_reviews(df, df.week_day, 'review_distribution_per_day', 'Day')
    analyze_reviews(df, df.month, 'review_distribution_per_month', 'Month')
    analyze_reviews(df, df.year, 'review_distribution_per_year', 'Year')

def export_week_day(df):
    for i in range(1,6):
        print(i, df.loc[df['overall']==i].groupby(['week_day']).size())

def export_month(df):
        for i in range(1,6):
            print(i, df.loc[df['overall']==i].groupby(['month']).size().values.tolist())

def export_year(df):
        for i in range(1,6):
            print(i, df.loc[df['overall']==i].groupby(['year']).size().values.tolist())
