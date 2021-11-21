# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 17:02:04 2021

@author: ivan
"""

import pandas as pd
import numpy as np   
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values

regressor = DecisionTreeRegressor(random_state=1)
regressor.fit(np.array(X).reshape(-1,1),np.array(y).reshape(-1,1))
print(regressor.predict([[6.5]]))