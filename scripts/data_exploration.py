# -*- coding: utf-8 -*-

### Import libraries ###

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style(style="darkgrid")

### Functions ###

def most_reviewed_products(df, n_products):
    reviews_per_product = df['asin'].value_counts()
    most_reviews = reviews_per_product.nlargest(n_products)
    most_reviews = most_reviews.reset_index()
    most_reviews = most_reviews.drop('asin', axis=1)
    definitive = df.merge(most_reviews, left_on='asin', right_on='index')
    definitive = definitive.drop('index', axis=1)
    return definitive


def most_active_reviewers(df, n_reviewers):
    n_reviews = df['reviewerID'].value_counts()
    most_reviews = n_reviews.nlargest(n_reviewers)
    most_reviews = most_reviews.reset_index()
    most_reviews = most_reviews.drop('reviewerID', axis=1)
    definitive = df.merge(most_reviews, left_on='reviewerID', right_on='index')
    definitive = definitive.drop('index', axis=1)
    return definitive


def analyze_reviews(df, df_attribute, title, name_file):
    print("Shape of df: ", df.shape)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.countplot(df_attribute, ax=ax)
    ax.set_title(title)
    ax.figure.savefig(r'./figures/1_{0}.svg'.format(name_file), format='svg')
    print('Exported 1_{}.svg'.format(name_file))
    #plt.show()
    # ax1.figure.savefig('figures/unverified_scoredistribution.svg', format='svg')
    # print("Exported unverified_scoredistribution.svg")

    print("Average Score: ", np.mean(df.overall))
    print("Median Score: ", np.median(df.overall))


def plot_boxplot_words(column, title, name_file):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.boxplot(column)
    ax.set_title(title)
    ax.figure.savefig(r'./figures/1_{0}.svg'.format(name_file), format='svg')
    print('Exported 1_{}.svg'.format(name_file))

#%%

def run(df):
    # Number of words for each review
    #plot_boxplot_words(df['n_words'], 'Reviews', 'lengthreviews')
    
    analyze_reviews(df, df.overall, 'Score distribution', 'score_distribution')
    '''
    # Score Distribution
    fig, ax1 = plt.subplots()
    sns.countplot(df.overall, ax=ax1)
    ax1.set_title('Score distribution')
    # plt.show()
    ax1.figure.savefig(r'./figures/1_score_distribution.svg', format='svg')
    print("Exported 1_score_distribution.svg")
    '''
    #Opinion Distribution
    analyze_reviews(df, df.opinion, 'Opinion distribution', 'opinion_distribution')
    '''
    fig, ax2 = plt.subplots()
    sns.countplot(df.opinion, ax=ax2)
    ax2.set_title('Opinion distribution')
    ax2.figure.savefig(r'./figures/1_opiniondistribution.svg', format='svg')
    print("Exported 1_opinion_distribution.svg")
    '''

    #Stacked barplot (x-axis asin code, y-axis opinion)
    fig, ax3 = plt.subplots(figsize=(15, 15))
    top_products = most_reviewed_products(df, 20)
    r = list(top_products['asin'].unique())
    positive = list(top_products.loc[top_products['opinion'] == 'positive', 'asin'].value_counts().reindex(top_products['asin'].unique(), fill_value=0))
    neutral = list(top_products.loc[top_products['opinion'] == 'neutral', 'asin'].value_counts().reindex(top_products['asin'].unique(), fill_value=0))
    negative = list(top_products.loc[top_products['opinion'] == 'negative', 'asin'].value_counts().reindex(top_products['asin'].unique(), fill_value=0))
    raw_data = {'positive': positive, 'neutral': neutral, 'negative': negative}
    raw_data = pd.DataFrame(raw_data)

    #print("Opinions ",raw_data)
    
    totals = [i+j+k for i,j,k in zip(raw_data['positive'], raw_data['neutral'], raw_data['negative'])]
    #totals = list(top_products['asin'].value_counts().reindex(top_products['asin'].unique(), fill_value=0))
    positive_percentage = [i / j * 100 for i, j in zip(raw_data['positive'], totals)]
    neutral_percentage = [i / j * 100 for i, j in zip(raw_data['neutral'], totals)]
    negative_percentage = [i / j * 100 for i, j in zip(raw_data['negative'], totals)]

    bar_width = 0.85

    ax3.bar(r, positive_percentage, color='#b5ffb9', edgecolor='white', width=bar_width, label='positive')
    ax3.bar(r, neutral_percentage, bottom=positive_percentage, color='#f9bc86', edgecolor='white', width=bar_width, label='neutral')
    ax3.bar(r, negative_percentage, bottom=[i + j for i, j in zip(positive_percentage, neutral_percentage)], color='#a3acff', edgecolor='white', width=bar_width, label='negative')
    #ax.xticks(r, names, rotation=90)
    ax3.set_xticklabels(r, rotation=90)
    ax3.set_xlabel('Code product')
    legend = ax3.legend(loc='best', shadow=True, fontsize='large')
    legend.get_frame().set_facecolor('#00FFCC')
    ax3.set_title('Opinion for besteller products')
    # plt.show()
    ax3.figure.savefig(r'./figures/1_sentiment_reviews_bestseller_products.svg', format='svg')
    print("Exported 1_sentiment_reviews_besteller_products.svg")

    # Top 50 reviewers
    fig, ax4 = plt.subplots(figsize=(15, 15))
    top_reviewers = most_active_reviewers(df, 50)
    '''
    top_reviewers = top_reviewers.groupby('reviewerID').agg({'overall':'mean',
                                                             'vote':'mean',
                                                             'asin':'count'}).sort_values(['asin'], ascending=[False]).reset_index()
    '''
    sns.countplot(top_reviewers.reviewerID, ax=ax4, order=top_reviewers['reviewerID'].value_counts().index)
    r = list(top_reviewers['reviewerID'].unique())
    ax4.set_xticklabels(r, rotation=90)
    ax4.set_title('Reviewers with most reviews')
    ax4.figure.savefig(r'./figures/1_reviewers_most_reviews.svg', format='svg')
    '''
    r = list(top_reviewers['reviewerID'].unique())
    bar_width = 0.85

    ax4.bar(r, top_reviewers['asin'], color='#f9bc86', edgecolor='white', width=bar_width)
    ax4.bar(r, top_reviewers['overall'], color='#b5ffb9', edgecolor='white', width=bar_width)
    ax4.set_xticklabels(r, rotation=90)
    ax4.set_xlabel('Reviewer ID')
    ax4.figure.savefig('figures/1_reviewers.svg', format='svg')
    print("Exported 1_reviewers.svg")
    '''
    #Non verified reviews
    print("### Unverified reviews exploration ###")
    unverified = df[df['verified'] == False]
    analyze_reviews(unverified, df.overall, "Unverified score distribution", 'unverified_score_distribution')

    #Verified reviews
    print("### Verified reviews exploration ###")
    verified = df[df['verified'] == True]
    analyze_reviews(verified, df.overall, "Verified score distribution", 'verified_score_distribution')

    reduced_df = df.copy()
    reduced_df = reduced_df[reduced_df['n_words'] <= 1000]
    reduced_unverified = reduced_df[reduced_df['verified'] == False]
    plot_boxplot_words(reduced_unverified['n_words'], 'Distribution of words in unverified reviews', 'length_unverified_reviews')
    #plot_boxplot_words(reduced_unverified['vote'], 'Unverified votes', 'unverified_votes')

    reduced_verified = reduced_df[reduced_df['verified'] == True]
    plot_boxplot_words(reduced_verified['n_words'], 'Distribution of words in verified reviews', 'length_verified_reviews')
    #plot_boxplot_words(reduced_verified['vote'], 'Verified votes', 'verified_votes')
    #plt.show()
    
    fig, ax5 = plt.subplots()
    ax5 = sns.violinplot(x=reduced_df['opinion'], y=reduced_df['n_words'])
    ax5.set_title('Distribution of words in review for each opinion')
    ax5.figure.savefig(r'./figures/1_correlation_words_opinion.svg', format='svg')
    
    analyze_reviews(df, df.week_day, 'Review distribution per day', 'review_distribution_per_day')
    unverified2 = unverified[unverified['opinion'] == 'negative']
    print("Negative unverified", unverified2.shape)
    unverified2_head = unverified2.head(10)['reviewText']
    for x in unverified2_head:
        print("Review: ", x)

    unverified3 = unverified[unverified['opinion'] == 'positive']
    print("Positive unverified", unverified3.shape)
    unverified3_head = unverified3.head(10)['reviewText']
    for x in unverified3_head:
        print("Review: ", x)
        

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
    #ax.xticks(r, names, rotation=90)
    ax6.set_xticklabels(r, rotation=90)
    ax6.set_xlabel('Reviewer ID')
    legend = ax6.legend(loc='best', shadow=True, fontsize='large')
    legend.get_frame().set_facecolor('#00FFCC')
    ax6.set_title('Opinion of top reviewers')
    plt.show()
    ax6.figure.savefig(r'./figures/1_opinion_top_reviewers.svg', format='svg')
    print("Exported 1_opinion_top_reviewers.svg")
    
    
    fig, ax7 = plt.subplots(figsize=(15, 15))
    #top_reviewers = most_active_reviewers(df, 50)
    r = list(top_reviewers['reviewerID'].unique())
    verified = list(top_reviewers.loc[top_reviewers['verified'] == True, 'reviewerID'].value_counts().reindex(top_reviewers['reviewerID'].unique(), fill_value=0))
    unverified = list(top_reviewers.loc[top_reviewers['verified'] == False, 'reviewerID'].value_counts().reindex(top_reviewers['reviewerID'].unique(), fill_value=0))
    #negative = list(top_reviewers.loc[top_reviewers['opinion'] == 'negative', 'reviewerID'].value_counts().reindex(top_reviewers['reviewerID'].unique(), fill_value=0))
    raw_data = {'verified': verified, 'unverified': unverified}
    raw_data = pd.DataFrame(raw_data)

    #print("Opinions ",raw_data)
    
    totals = [i+j for i,j in zip(raw_data['verified'], raw_data['unverified'])]
    #totals = list(top_products['asin'].value_counts().reindex(top_products['asin'].unique(), fill_value=0))
    verified_percentage = [i / j * 100 for i, j in zip(raw_data['verified'], totals)]
    unverified_percentage = [i / j * 100 for i, j in zip(raw_data['unverified'], totals)]
    #negative_percentage = [i / j * 100 for i, j in zip(raw_data['negative'], totals)]

    bar_width = 0.85

    ax7.bar(r, verified_percentage, color='#b5ffb9', edgecolor='white', width=bar_width, label='verified')
    ax7.bar(r, unverified_percentage, bottom=verified_percentage, color='#f9bc86', edgecolor='white', width=bar_width, label='unverified')
    #ax6.bar(r, negative_percentage, bottom=[i + j for i, j in zip(positive_percentage, neutral_percentage)], color='#a3acff', edgecolor='white', width=bar_width)
    #ax.xticks(r, names, rotation=90)
    ax7.set_xticklabels(r, rotation=90)
    ax7.set_xlabel('Reviewer ID')
    legend = ax7.legend(loc='best', shadow=True, fontsize='large')
    legend.get_frame().set_facecolor('#00FFCC')
    #ax6.legend(loc='lower left', fontsize='large')
    ax7.set_title('Verified vs Unverified reviews of top reviewers')
    plt.show()
    ax7.figure.savefig(r'./figures/1_verified_unverified.svg', format='svg')
    print("Exported 1_verified_unverified.svg")
    # Ordinamento date
    #df = df.sort_values('month_year', ascending=True)