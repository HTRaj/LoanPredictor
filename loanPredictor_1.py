# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 19:33:13 2019

@author: THILAKRAJ SHETTY
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

train = pd.read_csv('train.csv')
X_train = train.iloc[:, 0:12]
y_train = train.iloc[:, 12]
test = pd.read_csv('test.csv')
X_test = test.iloc[:, 0:12]

X_train['Gender'].fillna(X_train['Gender'].mode()[0], inplace = True)
X_train['Married'].fillna(X_train['Married'].mode()[0], inplace = True)
X_train['Dependents'].fillna(X_train['Dependents'].mode()[0], inplace = True)
X_train['Self_Employed'].fillna(X_train['Self_Employed'].mode()[0], inplace = True)
X_train['Credit_History'].fillna(X_train['Credit_History'].mode()[0], inplace = True)
X_train['LoanAmount'].fillna(128, inplace = True)
X_train['Loan_Amount_Term'].fillna(360, inplace = True)

X_test['Gender'].fillna(X_train['Gender'].mode()[0], inplace = True)
X_test['Married'].fillna(X_train['Married'].mode()[0], inplace = True)
X_test['Dependents'].fillna(X_train['Dependents'].mode()[0], inplace = True)
X_test['Self_Employed'].fillna(X_train['Self_Employed'].mode()[0], inplace = True)
X_test['Credit_History'].fillna(X_train['Credit_History'].mode()[0], inplace = True)
X_test['LoanAmount'].fillna(128, inplace = True)
X_test['Loan_Amount_Term'].fillna(360, inplace = True)


'''
X_train.apply(lambda x: len(x.unique()))
X_train.isnull().sum()
X_train['LoanAmount'].mean()
X_train.apply(lambda x: len(x.unique()))
X_test.isnull().sum()
'''

#Removing Unnessasry Column
X_train = X_train.drop('Loan_ID', axis = 1)
X_test = X_test.drop('Loan_ID', axis = 1)

X_train = X_train.values
X_test = X_test.values



from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_1 = LabelEncoder()
labelencoder_1.fit(X_train[:,0])
X_train[:, 0] = labelencoder_1.transform(X_train[:,0])
X_test[:, 0] = labelencoder_1.transform(X_test[:,0])

labelencoder_2 = LabelEncoder()
labelencoder_2.fit(X_train[:,1])
X_train[:, 1] = labelencoder_2.transform(X_train[:,1])
X_test[:, 1] = labelencoder_2.transform(X_test[:,1])

labelencoder_3 = LabelEncoder()
labelencoder_3.fit(X_train[:,2])
X_train[:, 2] = labelencoder_3.transform(X_train[:,2])
X_test[:, 2] = labelencoder_3.transform(X_test[:,2])

labelencoder_4 = LabelEncoder()
labelencoder_4.fit(X_train[:,3])
X_train[:, 3] = labelencoder_4.transform(X_train[:,3])
X_test[:, 3] = labelencoder_4.transform(X_test[:,3])

labelencoder_5 = LabelEncoder()
labelencoder_5.fit(X_train[:,4])
X_train[:, 4] = labelencoder_5.transform(X_train[:,4])
X_test[:, 4] = labelencoder_5.transform(X_test[:,4])

labelencoder_6 = LabelEncoder()
labelencoder_6.fit(X_train[:,10])
X_train[:, 10] = labelencoder_6.transform(X_train[:,10])
X_test[:, 10] = labelencoder_6.transform(X_test[:,10])


onehotencoder_1 = OneHotEncoder(categorical_features = [2])
onehotencoder_1.fit(X_train)
X_train = onehotencoder_1.transform(X_train).toarray()
X_test = onehotencoder_1.transform(X_test).toarray()
X_train = X_train[:, 1:]
X_test = X_test[:, 1:]

onehotencoder_2 = OneHotEncoder(categorical_features = [12])
onehotencoder_2.fit(X_train)
X_train = onehotencoder_2.transform(X_train).toarray()
X_test = onehotencoder_2.transform(X_test).toarray()
X_train = X_train[:, 1:]
X_test = X_test[:, 1:]

labelencoder_7 = LabelEncoder()
labelencoder_7.fit(y_train)
y_train = labelencoder_7.transform(y_train)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.cross_validation import cross_val_score
accuracies = cross_val_score(estimator = classifier, X=X_train, y = y_train, cv =10, n_jobs = -1)
accuracies.mean()

































