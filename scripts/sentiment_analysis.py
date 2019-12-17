# -*- coding: utf-8 -*-
from sentiment_data_preparation import sentiment_analysis_data_preparation
from sentiment_data_preparation import retrieve_opinion
from sentiment_data_preparation import vectorization
from sentiment_data_preparation import plot_frequency
from sentiment_data_preparation import zipf_law
from sentiment_data_preparation import token_frequency

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split   
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def train_predict_model(classifier, train_features, train_labels, test_features):
    # build model    
    classifier.fit(train_features, train_labels)
    # predict using model
    predictions = classifier.predict(test_features) 
    return predictions    

'''
def get_metrics(true_labels, predicted_labels):
    
    print('Accuracy:', np.round(
                        metrics.accuracy_score(true_labels, 
                                               predicted_labels),
                        4))
    print('Precision:', np.round(
                        metrics.precision_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'),
                        4))
    print('Recall:', np.round(
                        metrics.recall_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'),
                        4))
    print('F1 Score:', np.round(
                        metrics.f1_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'),
                        4))
'''
    

def plot_confusion_matrix(cm, name_model, classes=['positive', 'negative']):
    fig, ax = plt.subplots(figsize=(10,10))
    img = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    title = 'Confusion matrix ' + name_model
    ax.set_title(title)
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
    ax.figure.savefig('figures/2_confusion_matrix_{}.svg'.format(name_model),
                      format='svg')
    print('Exported 2_confusion_matrix.svg')



def compute_confusion_matrix(true_labels, predicted_labels, classes=['positive', 'negative']):
    cm = metrics.confusion_matrix(y_true=true_labels, y_pred=predicted_labels, 
                                  labels=classes)
    return cm

'''
def display_confusion_matrix(true_labels, predicted_labels, classes=['positive', 'negative']):
    
    total_classes = len(classes)
    level_labels = [total_classes*[0], list(range(total_classes))]

    cm = compute_confusion_matrix(true_labels=true_labels, 
                                  predicted_labels=predicted_labels,
                                  classes=classes)
    cm_frame = pd.DataFrame(data=cm, 
                            columns=pd.MultiIndex(levels=[['Predicted:'], classes], 
                                                  codes=level_labels), 
                            index=pd.MultiIndex(levels=[['Actual:'], classes], 
                                                codes=level_labels)) 
    print(cm_frame) 
    return cm
    
    
def display_classification_report(true_labels, predicted_labels, classes=['positive', 'negative']):

    report = metrics.classification_report(y_true=true_labels, 
                                           y_pred=predicted_labels, 
                                           labels=classes) 
    print(report
'''
    
'''
def display_model_performance_metrics(true_labels, predicted_labels, classes=['positive', 'negative']):
    print('Model Performance metrics:')
    print('-'*30)
    get_metrics(true_labels=true_labels, predicted_labels=predicted_labels)
    print('\nModel Classification report:')
    print('-'*30)
    display_classification_report(true_labels=true_labels, predicted_labels=predicted_labels, 
                                  classes=classes)
    
    print('\nPrediction Confusion Matrix:')
    print('-'*30)
    display_confusion_matrix(true_labels=true_labels, predicted_labels=predicted_labels, 
                             classes=classes)
'''

def run(df):
    current_directory = os.getcwd()
    os.chdir('..')
    
    df = sentiment_analysis_data_preparation(df)
    retrieve_opinion(df, 'positive')
    retrieve_opinion(df, 'negative')
    cv = CountVectorizer(max_features=10000, min_df=7, max_df=0.8)
    term_frequency = vectorization(df, cv)   
    plot_frequency(term_frequency)
    zipf_law(term_frequency)
    token_frequency(term_frequency, 'positive')
    token_frequency(term_frequency, 'negative')
    
    
    # Machine learning
    # Valutare di cancellare anche reviews con tot parole
    reviews = np.array(df['preprocessedReview'])
    sentiments = np.array(df['opinion']) 
    reviews_train, reviews_test, sentiment_train, sentiment_test = train_test_split(reviews, 
                                                                                    sentiments, 
                                                                                    test_size=0.2, 
                                                                                    random_state=42)
    cv_train_features = cv.fit_transform(reviews_train)
    cv_test_features = cv.transform(reviews_test)


    lr = LogisticRegression(penalty='l2', max_iter=100, C=1)
    svc = LinearSVC()
    # Logistic regression su BOW
    lr_predictions_bow = train_predict_model(classifier=lr,
                                             train_features=cv_train_features, 
                                             train_labels=sentiment_train, 
                                             test_features=cv_test_features)
    '''
    display_model_performance_metrics(true_labels=sentiment_test, 
                                      predicted_labels=lr_predictions_bow)
    '''
    cm = compute_confusion_matrix(true_labels=sentiment_test,
                                  predicted_labels=lr_predictions_bow)
    
    plot_confusion_matrix(cm, 'Logistic Regression')
    
    
    
    # SVM su BOW
    svc_predictions_bow = train_predict_model(classifier=svc,
                                              train_features=cv_train_features, 
                                              train_labels=sentiment_train, 
                                              test_features=cv_test_features)
    
    cm = compute_confusion_matrix(true_labels=sentiment_test,
                                  predicted_labels=svc_predictions_bow)
    
    plot_confusion_matrix(cm, 'SVM')
    '''
    display_model_performance_metrics(true_labels=sentiment_test,
                                      predicted_labels=svc_predictions_bow)
    '''
    os.chdir(current_directory)
    
    '''
        
    tv = TfidfVectorizer(min_df=7, max_df=0.8, ngram_range=(1, 2),
                     sublinear_tf=True)
    tv_train_features = tv.fit_transform(reviews_train)
    tv_test_features = tv.transform(reviews_test)
    
    # Logistic regression su TFIDF
    lr_predictions_tfidf = train_predict_model(classifier=lr,
                                               train_features=tv_train_features, 
                                               train_labels=sentiment_train, 
                                               test_features=tv_test_features)
    display_model_performance_metrics(true_labels=sentiment_test, 
                                      predicted_labels=lr_predictions_tfidf)
    # SVM su TFIDF
    svc_predictions_tfidf = train_predict_model(classifier=svc,
                                                train_features=tv_train_features, 
                                                train_labels=sentiment_train, 
                                                test_features=tv_test_features)
    display_model_performance_metrics(true_labels=sentiment_test,
                                      predicted_labels=svc_predictions_tfidf)
    
    # Plot confusion matrix
    cm = compute_confusion_matrix(true_labels=sentiment_test,
                                  predicted_labels=svc_predictions_tfidf)
    

    plot_confusion_matrix(cm)
    '''
    

    
    
    
    
    