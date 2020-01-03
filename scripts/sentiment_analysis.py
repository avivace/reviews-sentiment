# -*- coding: utf-8 -*-
from sentiment_data_preparation import sentiment_analysis_data_preparation
from sentiment_data_preparation import retrieve_opinion
from sentiment_data_preparation import get_term_frequency
from sentiment_data_preparation import plot_frequency
from sentiment_data_preparation import zipf_law
from sentiment_data_preparation import token_frequency


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report


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
    #os.chdir('..')
    
    df = sentiment_analysis_data_preparation(df)
    retrieve_opinion(df, 'positive')
    retrieve_opinion(df, 'negative')
    count_vector = CountVectorizer() #max_features=10000, min_df=7, max_df=0.8)

    count_vector2 = CountVectorizer() #Used only to calculate the term frequency on the dataset
    term_frequency = get_term_frequency(df, count_vector)
    plot_frequency(term_frequency)
    zipf_law(term_frequency)
    token_frequency(term_frequency, 'positive')
    token_frequency(term_frequency, 'negative')
    
    
    ### Machine learning ###
    reviews = np.array(df['preprocessedReview'])
    sentiments = np.array(df['opinion'])

    #Simple train/test split
    '''reviews_train, reviews_test, sentiment_train, sentiment_test = train_test_split(reviews,
                                                                                    sentiments, 
                                                                                    test_size=0.2, 
                                                                                    random_state=42)
    cv_train_features = count_vector.fit_transform(reviews_train)
    cv_test_features = count_vector.transform(reviews_test)
    print(reviews_train.shape)
    print(reviews_test.shape)
    print(sentiment_train.shape)
    print(sentiment_test.shape)
    print("Words in BOW: ", len(count_vector.vocabulary_))'''

    
    # Logistic regression su BOW
    '''
    lr = LogisticRegression(penalty='l2', max_iter=4000, C=1)
    lr_predictions_bow = train_predict_model(classifier=lr,
                                             train_features=cv_train_features, 
                                             train_labels=sentiment_train, 
                                             test_features=cv_test_features)'''
    '''
    display_model_performance_metrics(true_labels=sentiment_test, 
                                      predicted_labels=lr_predictions_bow)
    '''
    '''cm = compute_confusion_matrix(true_labels=sentiment_test,
                                  predicted_labels=lr_predictions_bow)
    
    plot_confusion_matrix(cm, 'Logistic Regression')'''
    
    
    
    # SVM su BOW
    ''' svc = LinearSVC()
    svc_predictions_bow = train_predict_model(classifier=svc,
                                              train_features=cv_train_features, 
                                              train_labels=sentiment_train, 
                                              test_features=cv_test_features)
    
    cm = compute_confusion_matrix(true_labels=sentiment_test,
                                  predicted_labels=svc_predictions_bow)
    
    plot_confusion_matrix(cm, 'SVM')'''
    '''
    display_model_performance_metrics(true_labels=sentiment_test,
                                      predicted_labels=svc_predictions_bow)
    '''

    # Logistic Regression CV with grid search su BOW
    reviews_train, reviews_validation, sentiment_train, sentiment_validation = train_test_split(reviews,
                                                                                    sentiments,
                                                                                    test_size=0.5,
                                                                                    random_state=42)
    count_vector_features = count_vector.fit_transform(reviews_train)
    count_vector_validation_features = count_vector.transform(reviews_validation)

    param_grid = [
        {
            'C': np.logspace(0, 4, 4)
        }
    ]

    # Create grid search object
    lr = LogisticRegression(max_iter=10000)
    lr_grid = GridSearchCV(lr, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
    # Fit on data
    best_lr = lr_grid.fit(count_vector_features, sentiment_train)
    print("Best params")
    for i in best_lr.best_params_:
        print(i, best_lr.best_params_[i])

    y_true, y_pred = sentiment_validation, best_lr.predict(count_vector_validation_features)
    print("Report on validation set")
    print(classification_report(y_true, y_pred))

    fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred, pos_label='positive') #TODO QUI DA' ERRORE!
    roc_auc = metrics.auc(fpr, tpr)

    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    #SVC CV with grid search su BOW
    '''param_grid = [
        {
            'C':np.arange(0.01,100,10)
        }
    ]

    # Create grid search object
    svc = LinearSVC()
    lr = GridSearchCV(svc, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
    # Fit on data
    best_lr = lr.fit(count_vector_features, sentiments)
    print(sorted(best_lr.cv_results_.keys()))'''



    '''     
    #TFIDF Learning
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







