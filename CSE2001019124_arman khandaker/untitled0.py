# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 12:01:56 2023

@author: Dell
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

dataset = pd.read_csv('Social_Network_Ads.csv')

x = dataset.iloc[:,[2,3]].values

y = dataset.iloc[:,4].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

model = GaussianNB()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)


print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("F1 score:", f1)