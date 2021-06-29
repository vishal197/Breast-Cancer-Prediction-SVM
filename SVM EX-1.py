# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 18:34:42 2021

@author: Vishal pc
"""
import pandas as pd
import numpy as np
import seaborn as sns
import os

#1.	Load thedata into a pandas dataframe named data_firstname where first name is you name
filename = 'breast_cancer.csv'
path = 'E:/Centennial/semester-2/Supervised Learning/SVM'
fullpath = os.path.join(path,filename)
data_Vishal = pd.read_csv(fullpath)

#2.
#a.	Check the names and types of columns.
data_Vishal.dtypes

#b.	Check the missing values
data_Vishal.isnull().sum()

#c.	Check the statistics of the numeric fields (mean, min, max, median, count..etc.)
data_Vishal.mean()
data_Vishal.min()
data_Vishal.max()
data_Vishal.median()
data_Vishal.count()

#3.	Replace the ‘?’ mark in the ‘bare’ column by np.nan and change the type to ‘float’
data_Vishal['bare'] = data_Vishal['bare'].replace('?', np.nan)
data_Vishal['bare']=data_Vishal['bare'].astype(np.float64)
data_Vishal['bare'].dtypes

#4.	Fill any missing data with the median of the column
data_Vishal['bare'].fillna(data_Vishal['bare'].median(), inplace = True)

#5.	Drop the ID column
data_Vishal = data_Vishal.drop(columns='ID', axis=1)

#6.	Using Pandas, Matplotlib, seaborn (you can use any or a mix) generate 3-5 plots
# and add them to your written response explaining what are the key insights and findings from the plots.

sns.boxplot(y=data_Vishal['thickness'], x=data_Vishal['class'])
sns.boxplot(y=data_Vishal['shape'], x=data_Vishal['class'])
sns.boxplot(y=data_Vishal['Epith'], x=data_Vishal['class'])
sns.boxplot(y=data_Vishal['size'], x=data_Vishal['class'])

#7.	Separate the features from the class
data_features = data_Vishal[data_Vishal.columns.difference(['class'])]
data_target = data_Vishal['class']

#8.	Split your data into train 80% train and 20% test, use the last two digits of your student number for the seed
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, test_size=0.2, random_state=2) #last two digits of stu id is 02 but 02 is not accepted


#Build Classification Models

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#function for building a model
def svcModel(kernelType=None):
    if kernelType == None:
        kernelType = 'linear'
    
    if kernelType == 'linear':
        classifierSVC = SVC(kernel=kernelType, C=0.1)
    else:
        classifierSVC = SVC(kernel=kernelType)
    
    classifierSVC.fit(X_train, y_train)
    
    prediction_X_train = classifierSVC.predict(X_train)
    accuracy_X_train = accuracy_score(y_train, prediction_X_train)
    
    prediction_X_test = classifierSVC.predict(X_test)
    accuracy_X_test = accuracy_score(y_test, prediction_X_test)
    
    print('Model kernel: ',kernelType)
    print('Accuracy for training data is: ', accuracy_X_train * 100)
    print('Accuracy for testing data is: ', accuracy_X_test * 100)
    
    print('Confusion matrix for training data: ')
    print(confusion_matrix(y_train, prediction_X_train))
    print('Confusion matrix for testing data: ')
    print(confusion_matrix(y_test, prediction_X_test))
    
kernelType = ['linear', 'rbf', 'poly', 'sigmoid']

for type in kernelType:
    svcModel(type)
