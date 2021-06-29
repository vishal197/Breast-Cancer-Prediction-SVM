# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 21:32:59 2021

@author: Vishal pc
"""
import pandas as pd
import numpy as np
import os

#1.	Load thedata into a pandas dataframe named data_firstname where first name is you name
filename = 'breast_cancer.csv'
path = 'E:/Centennial/semester-2/Supervised Learning/SVM'
fullpath = os.path.join(path,filename)
data_Vishal_df2 = pd.read_csv(fullpath)

#2.	Replace the ‘?’ mark in the ‘bare’ column by np.nan and change the type to ‘float’
data_Vishal_df2['bare'] = data_Vishal_df2['bare'].replace('?', np.nan)
data_Vishal_df2['bare'] = data_Vishal_df2['bare'].astype(np.float64)
data_Vishal_df2.dtypes

#3.	Drop the ID column
data_Vishal_df2 = data_Vishal_df2.drop(columns='ID', axis=1)

#4.	Separate the features from the class
data_features = data_Vishal_df2[data_Vishal_df2.columns.difference(['class'])]
data_target = data_Vishal_df2['class']

#5.	Split your data into train 80% train and 20% test   use the last two digits of your student number for the seed. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, test_size=0.2, random_state=2)

#6.	Using the preprocessing library to define two transformer objects to transform your training data
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline

num_pipe_Vishal = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('scaler', StandardScaler())
    ])

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#8
pipe_svm_Vishal = Pipeline([
        ("pipe-1", num_pipe_Vishal),
        ("svm", SVC(random_state=2))
    ])

#10
param_grid_svm = {
        'svm__kernel': ['linear', 'rbf','poly'],
        'svm__C':  [0.01,0.1, 1, 10, 100],
        'svm__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
        'svm__degree':[2,3]
    }

#12
from sklearn.model_selection import GridSearchCV
grid_search_Vishal = GridSearchCV(
        estimator=pipe_svm_Vishal,
        param_grid=param_grid_svm,
        scoring='accuracy',
        refit=True,
        verbose=3
    )

#14.Fit your training data to the gird search object
grid_search_Vishal.fit(X_train, y_train)

#15.Print out the best parameters and note it in your written response
best_parameters = grid_search_Vishal.best_params_
print(best_parameters)
#16.Print out the best estimator and note it in your written response
grid_search_Vishal.best_estimator_

#17.Predict the test data using the fine-tuned model identified during grid search
prediction_X_test = grid_search_Vishal.best_estimator_.predict(X_test)

#18.Printout the accuracy score and note it in your written response
accuracy_X_test = accuracy_score(y_test, prediction_X_test)
accuracy_X_test * 100

#19.Create an object that holds the best model 
best_model_Vishal = grid_search_Vishal.best_estimator_

#20.Save the model using the joblib
import joblib
joblib.dump(best_model_Vishal, "best_model_Vishal.pkl")

#21.Save the full pipeline using the joblib 
joblib.dump(pipe_svm_Vishal, "full_pipeline.pkl")
