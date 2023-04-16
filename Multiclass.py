# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 21:22:45 2023

@author: adityabn
"""

# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# load data
train_data = pd.read_csv('C:/Users/adityabn/Downloads/train_python_base.csv')
test_data = pd.read_csv('C:/Users/adityabn/Downloads/test_python_base.csv')



# separate features and labels
train_features = train_data.iloc[:, 2:].values
train_labels = train_data.iloc[:, 1].values
train_features = train_data.loc[:,selected_covariates].values

test_features = test_data.iloc[:, 1:].values
test_features = test_data.loc[:,selected_covariates].values
train_labels = train_labels-1

# split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.2, random_state=42, )

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=2)
# define classifiers and their hyperparameters
classifiers = [
    {'name': 'SVM',
     'classifier': SVC(random_state=42),
     'params': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'sigmoid']}
    },
    {'name': 'Random Forest',
     'classifier': RandomForestClassifier(random_state=42),
     'params': {'n_estimators': [50, 100, 200,500], 'max_depth': [5, 10, None]}
    }
]

# iterate over classifiers and fine-tune hyperparameters
for clf in classifiers:
    print('Classifier:', clf['name'])
    grid = GridSearchCV(clf['classifier'], clf['params'], cv=10)
    grid.fit(train_features, train_labels)
    print('Best hyperparameters:', grid.best_params_)
    model = grid.best_estimator_
    y_pred = model.predict(X_val)
    print('Validation set accuracy:', accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred))

svc = SVC(C=10,kernel='rbf')
svc.fit(train_features,train_labels)
y_pred = svc.predict(test_features)
y_pred +=1

scores = cross_val_score(svc, train_features, train_labels, cv=10) # cv=10 means 10-fold cross-validation
mean_accuracy = scores.mean()

with open("C:/Users/adityabn/Downloads/multiclass_Blogberry.txt", "w") as f:
    for elem in y_pred:
        f.write(str(elem) + "\n")