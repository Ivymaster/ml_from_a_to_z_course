# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 19:47:29 2021

@author: ivan
""" 

import pandas as pd
import numpy as np   
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values

regressor = RandomForestRegressor(n_estimators=20)
regressor.fit(np.array(X).reshape(-1,1),np.array(y).reshape(-1,1))
print(regressor.predict([[6.5]]))