# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 18:45:11 2021

@author: ivan
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("50_Startups.csv")

#separating dependent and independent variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#replace cathegorical data with binary vector
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [-1])], remainder="passthrough") 
X = np.array(ct.fit_transform(X))
 
#split training and test data
X_test, X_train, y_test, y_train = train_test_split(X, y,  test_size=0.2, random_state=1)

#create and train multiple linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)


#predict results with the new model, on the train data
y_predict = regressor.predict(X_test)

print(y_predict)
print(y_test)