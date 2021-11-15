# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 23:19:43 2021

@author: ivan
"""

import numpy as np
import matplotlib as plt
import pandas as pd 

dataset = pd.read_csv("Data.csv")

#independent variables
X = dataset.iloc[:, :-1].values 

#dependent varialbe
y = dataset.iloc[:, -1].values 


#replace missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X[:, 1:])
X[:, 1:] = imputer.transform(X[:, 1:])

#encode categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough")
X = np.array(ct.fit_transform(X))

#encode label data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print(y)