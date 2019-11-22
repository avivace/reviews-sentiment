# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
color = sns.color_palette()
sns.set_style("dark")

pathfile = r'C:\Users\luca\Documents\GitHub\reviews-sentiment\datasets'

df = pd.read_csv(pathfile + '\Gourmet_food.csv', index_col=0)

#%%
print("exploratory_data_analysis")
print("Number of reviews:", len(df))

# Score Distribution
ax = plt.axes()
sns.countplot(df.overall, ax=ax)
ax.set_title('Score Distribution')
plt.show()

print("Average Score: ", np.mean(df.overall))
print("Median Score: ", np.median(df.overall))
#%%

def most_reviewed_products(n_products):
    reviews_per_product = df['asin'].value_counts()
    most_reviews = reviews_per_product.nlargest(n_products)
    most_reviews = most_reviews.reset_index()
    most_reviews = most_reviews.drop('asin', axis=1)
    
    definitive = df.merge(most_reviews, left_on='asin', right_on='index')
    definitive = definitive.drop('index', axis=1)
    
    return definitive
    
top_products = most_reviewed_products(20)

#%% Stacked barplot (x-axis asin code, y-axis opinion)
### TODO: barre con difetto da fixare
r = list(top_products['asin'].unique())
positive = list(top_products.loc[top_products['opinion'] == 'positive', 'asin'].value_counts())
neutral = list(top_products.loc[top_products['opinion'] == 'neutral', 'asin'].value_counts())
negative = list(top_products.loc[top_products['opinion'] == 'negative', 'asin'].value_counts())
raw_data = {'positive': positive, 'neutral': neutral, 'negative': negative}

totals = list(top_products['asin'].value_counts())
positive_percentage = [i / j * 100 for i, j in zip(positive, totals)]
neutral_percentage = [i / j * 100 for i, j in zip(neutral, totals)]
negative_percentage = [i / j * 100 for i, j in zip(negative, totals)]

bar_width = 0.85
names = tuple(r)

plt.bar(r, positive_percentage, color='#b5ffb9', edgecolor='white', width=bar_width)
plt.bar(r, neutral_percentage, bottom=positive_percentage, color='#f9bc86', edgecolor='white', width=bar_width)
plt.bar(r, negative_percentage, bottom=[i + j for i, j in zip(positive_percentage, neutral_percentage)], color='#a3acff', edgecolor='white', width=bar_width)
plt.xticks(r, names)
plt.xlabel("group")
plt.show()

#%% TODO 
# Correlation between votes and opinion with a boxplot.
# Maybe a better representation can be considered

def significative_reviews(n_votes):
    return df[df['vote'] >= n_votes]

df_zero = significative_reviews(5)
# Use a color palette
plt.boxplot(x=df_zero['opinion'], y=pd.to_numeric(df_zero['vote'], errors='coerce'), palette="Blues")
plt.show()

#%%
# library & dataset
import seaborn as sns
df_i = sns.load_dataset('iris')
 
# Use a color palette
sns.boxplot(y=df_i["sepal_length"], palette="Blues")
#sns.plt.show()

#%%

ax = plt.axes()
sns.countplot(df.opinion, ax=ax)
ax.set_title('Sentiment Positive vs Negative Distribution')
plt.show()

print("Proportion of positive review:", len(df[df.opinion == "positive"]) / len(df))
print("Proportion of neutral review:", len(df[df.opinion == "neutral"]) / len(df))
print("Proportion of negative review:", len(df[df.opinion == "negative"]) / len(df))




