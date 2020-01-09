# -*- coding: utf-8 -*-

### Import libraries ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
import itertools
from pathlib import Path
from data_utils import text_preprocessing



figOutputPath = Path("../figures/")

### Functions ###

def plot_confusion_matrix(cm, name_img, classes=['negative', 'positive']):
    fig, ax = plt.subplots(figsize=(10,10))
    img = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    #ax.set_title('Confusion matrix {}'.format(title))
    ax.axis('off')
    fig.colorbar(img)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks, classes)
    ax.set_yticks(tick_marks, classes)
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.figure.savefig(figOutputPath / '2_confusion_matrix_{}.svg'.format(name_img),
                      format='svg')


def plot_roc(y_true, y_pred, name_img, pos_label=1):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred[:, 1], pos_label)
    roc_auc = metrics.auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(10,10))
    #ax.set_title('Receiver Operating Characteristic of {}'.format(title))
    ax.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    ax.legend(loc='lower right')
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    ax.figure.savefig(figOutputPath / '2_roc_{}.svg'.format(name_img),
                      format='svg')


def wordcloud(text, sentiment, title=None):
    wordcloud = WordCloud(
        background_color='whitesmoke',
        max_words=200,
        max_font_size=120,
        scale=3,
        random_state=42,
        width=1000,
        height=1000,
    ).generate(str(text))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('off')   
    ax.imshow(wordcloud, interpolation='nearest')
    ax.figure.savefig(figOutputPath / '2_wordcloud_{}.svg'.format(sentiment),
                      format='svg')
    print('Exported 2_wordcloud_{}.svg'.format(sentiment))
    
    
def retrieve_opinion(df, sentiment):
    opinion = df[df['opinion'] == sentiment]
    reviews = opinion['preprocessedReview'].tolist()
    wordcloud(reviews, sentiment)
    

def get_term_frequency(df, cvector):
    cvector.fit(df.preprocessedReview)
    
    negative_matrix = cvector.transform(df[df['opinion'] == 'negative']['preprocessedReview'])
    negative_words = negative_matrix.sum(axis=0)
    negative_frequency = [(word, negative_words[0, idx]) for word, idx in cvector.vocabulary_.items()]
    negative_tf = pd.DataFrame(list(sorted(negative_frequency, key = lambda x: x[1], reverse=True)),
                               columns=['Terms','negative'])
    negative_tf = negative_tf.set_index('Terms')
    
    positive_matrix = cvector.transform(df[df['opinion'] == 'positive']['preprocessedReview'])
    positive_words = positive_matrix.sum(axis=0)
    positive_frequency = [(word, positive_words[0, idx]) for word, idx in cvector.vocabulary_.items()]
    positive_tf = pd.DataFrame(list(sorted(positive_frequency, key = lambda x: x[1], reverse=True)),
                               columns=['Terms','positive'])
    positive_tf = positive_tf.set_index('Terms')
    
    term_frequency_df = pd.concat([negative_tf, positive_tf], axis=1)
    term_frequency_df['total'] = term_frequency_df['negative'] + term_frequency_df['positive']
    return term_frequency_df


def plot_frequency(df):
    #Frequency plot
    y_pos = np.arange(500)
    fig, ax = plt.subplots(figsize=(10,8))
    s = 1
    expected_zipf = [df.sort_values(by='total', ascending=False)['total'][0]/(i+1)**s for i in y_pos]
    ax.bar(y_pos, df.sort_values(by='total', ascending=False)['total'][:500], align='center', alpha=0.5)
    ax.plot(y_pos, expected_zipf, color='r', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Token')
    ax.set_yticks([0, 200000, 400000])
    ax.set_yticklabels(["0", "200K", "400K"])
    #ax.set_title('Top 500 tokens in reviews')
    ax.figure.savefig(figOutputPath / '2_plot_frequency.svg', format='svg')
    print('Exported 2_plot_frequency.svg')
    

def token_frequency(df, sentiment):
    y_pos = np.arange(50)
    plt.figure(figsize=(12,10))
    plt.bar(y_pos, df.sort_values(by=sentiment, ascending=False)[sentiment][:50], align='center', alpha=0.5)
    plt.xticks(y_pos, df.sort_values(by=sentiment, ascending=False)[sentiment][:50].index, rotation='vertical')
    plt.ylabel('Frequency')
    plt.xlabel('Token')
    #plt.title('Top 50 tokens in {} reviews'.format(sentiment))
    plt.savefig(figOutputPath / '2_token_frequency_{}.svg'.format(sentiment), format='svg')
    print('Exported 2_token_frequency_{}.svg'.format(sentiment))


def zipf_law(df):
    # Plot of absolute frequency
    from pylab import arange, argsort, loglog, logspace, log10, text
    counts = df.total
    tokens = df.index
    ranks = arange(1, len(counts)+1)
    indices = argsort(-counts)
    frequencies = counts[indices]
    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_ylim(1,10**6)
    ax.set_xlim(1,10**6)
    loglog(ranks, frequencies, marker=".")
    ax.plot([1,frequencies[0]],[frequencies[0],1],color='r')
    #ax.set_title("Zipf plot for phrases tokens")
    ax.set_xlabel("Frequency rank of token")
    ax.set_ylabel("Absolute frequency of token")
    ax.grid(True)
    for n in list(logspace(-0.5, log10(len(counts)-2), 15).astype(int)):
        dummy = text(ranks[n], frequencies[n], " " + tokens[indices[n]], 
                     verticalalignment="bottom",
                     horizontalalignment="left")
    ax.figure.savefig(figOutputPath / '2_zipf_law.png', format='png')
    print('Exported 2_zipf_law.png')
    

def undersampling(df):
    positive, negative = df.opinion.value_counts()
    df_positive = df[df.opinion == 'positive']
    df_positive = df_positive.sample(negative, random_state=42)
    df_negative = df[df.opinion == 'negative']
    df = pd.concat([df_positive, df_negative])
    df = df.sample(frac=1)
    return df


    
def run(df):   
    df.drop(df[df.opinion == 'neutral'].index, inplace=True)
    count_vector_exploration = CountVectorizer(max_features=10000, ngram_range=(1, 2))
    df['words'] = [len(t) for t in df['preprocessedReview']]
    df = df[df['words'] <= 300]
    df = df[df['words'] > 5]
    retrieve_opinion(df, 'positive')
    retrieve_opinion(df, 'negative')
    term_frequency = get_term_frequency(df, count_vector_exploration)
    zipf_law(term_frequency)
    plot_frequency(term_frequency)
    token_frequency(term_frequency, 'positive')
    token_frequency(term_frequency, 'negative')
    ### Machine learning ###
    df = undersampling(df)
    count_vector_sentiment = CountVectorizer(max_features=10000, ngram_range=(1, 2))
    reviews = np.array(df['preprocessedReview'])
    sentiments = np.array(df['opinion'])
    sentiments[sentiments == 'positive'] = 1
    sentiments[sentiments == 'negative'] = 0
    sentiments = sentiments.astype('int')

    # Logistic Regression CV with grid search su BOW
    reviews_train, reviews_validation, sentiment_train, sentiment_validation = train_test_split(reviews, sentiments, test_size=0.2, random_state=42)
    count_vector_features = count_vector_sentiment.fit_transform(reviews_train)
    count_vector_validation_features = count_vector_sentiment.transform(reviews_validation)

    param_grid = [
        {
            'C': np.logspace(0, 4, 4)
        }
    ]

    # Create grid search object
    lr = LogisticRegression(max_iter=10000, random_state=42)
    lr_grid = GridSearchCV(lr, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
    # Fit on data
    best_lr = lr_grid.fit(count_vector_features, sentiment_train)
    print("Best params")
    for i in best_lr.best_params_:
        print(i, best_lr.best_params_[i])

    y_true, y_pred, y_pred_roc = sentiment_validation, best_lr.predict(count_vector_validation_features), best_lr.predict_proba(count_vector_validation_features)
    print("Report on validation set")
    print(classification_report(y_true, y_pred))
    cm = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1])
    plot_confusion_matrix(cm, 'lr')
    plot_roc(y_true, y_pred_roc, 'lr')

    # Multinomial Bayes CV with grid search su BOW
    reviews_train, reviews_validation, sentiment_train, sentiment_validation = train_test_split(reviews, sentiments, test_size=0.2, random_state=42)
    count_vector_features = count_vector_sentiment.fit_transform(reviews_train)
    count_vector_validation_features = count_vector_sentiment.transform(reviews_validation)

    param_grid = [
        {
            'alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001)
        }
    ]

    # Create grid search object
    nb = MultinomialNB()
    nb_grid = GridSearchCV(nb, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
    # Fit on data
    best_nb = nb_grid.fit(count_vector_features, sentiment_train)

    print("Best params")
    for i in best_nb.best_params_:
        print(i, best_nb.best_params_[i])

    y_true, y_pred, y_pred_roc = sentiment_validation, best_nb.predict(count_vector_validation_features), best_nb.predict_proba(count_vector_validation_features)
    print("Report on validation set")
    print(classification_report(y_true, y_pred))
    cm = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1])
    plot_confusion_matrix(cm, 'nb')
    plot_roc(y_true, y_pred_roc, 'nb')

    return best_nb, best_lr, count_vector_sentiment


    

def compute_single(string, model, count_vector_sentiment):
    a = text_preprocessing([string], remove_less_frequent=False)
    a = np.asarray([ " ".join(a[0]) ])
    count_vector_validation_features2 = count_vector_sentiment.transform(a)
    return model.predict_proba(count_vector_validation_features2)