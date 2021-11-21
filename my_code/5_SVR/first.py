# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 15:48:18 2021

@author: ivan
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values

#apply feature scaling
sc_1 = StandardScaler()
sc_2 = StandardScaler()
 
X_scaled = sc_1.fit_transform(np.array(X).reshape(len(X), -1))
y_scaled = sc_2.fit_transform(np.array(y).reshape(len(y), -1))
 
#train SVR model
regressor = SVR(kernel="rbf")
regressor.fit(X_scaled, y_scaled)

#predict results
#input data needs to be scaled, and the output needs to be reverse scaled
y_predict = sc_2.inverse_transform(regressor.predict(sc_1.fit_transform(np.array(X).reshape(len(X), -1))))

plt.scatter(X, y, color="red")
plt.plot(X, y_predict, color="blue")