# -*- coding: utf-8 -*-
"""
Created on Tue May 12 22:48:42 2020

@author: skbar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv(r"C:\PythonCode\MachineLearning\Classifiers\data\Social_Network_Ads.csv")
X=data.iloc[:,[2,3]].values
y=data.iloc[:,4].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)


#as the data for different features are in different scale..it will be better to
#scale the same

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train= sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

#fitting  logistic model
from sklearn.linear_model import LogisticRegression
regressor= LogisticRegression(random_state=0)
regressor.fit(X_train,y_train)

y_pred= regressor.predict(X_test)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_pred)