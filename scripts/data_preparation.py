# -*- coding: utf-8 -*-

import pandas as pd

with open('DatasetsPath.txt', 'r') as myfile:
    pathfile = myfile.read()

#pathfile = r'C:\Users\Matteo\University\Magistrale\Data Analytics\Progetto\reviews-sentiment\datasets'
df = pd.read_json(pathfile + '\Grocery_and_Gourmet_Food_5.json', lines=True)


df = df.drop(['image',
              'reviewTime',
              'reviewerID',
              'reviewerName',
              'style',
              'unixReviewTime'], axis=1)

df['vote'].fillna(0, inplace=True)
df['vote'] = pd.to_numeric(df['vote'], errors='coerce')
df.dropna(inplace=True)

df.loc[df.overall == 3, 'opinion'] = "neutral"
df.loc[df.overall > 3, 'opinion'] = "positive"
df.loc[df.overall < 3, 'opinion'] = "negative"

df.to_csv(pathfile + '\Gourmet_food.csv')
