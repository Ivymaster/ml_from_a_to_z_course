# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 23:19:43 2021

@author: ivan
"""

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib as plt
import pandas as pd

dataset = pd.read_csv("Data.csv")

# independent variables
X = dataset.iloc[:, :-1].values

# dependent varialbe
y = dataset.iloc[:, -1].values


# replace missing data
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X[:, 1:])
X[:, 1:] = imputer.transform(X[:, 1:])

# encode categorical data
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough")
X = np.array(ct.fit_transform(X))

# encode label data
le = LabelEncoder()
y = le.fit_transform(y)

# split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

# feature scaling
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print(X_train)
