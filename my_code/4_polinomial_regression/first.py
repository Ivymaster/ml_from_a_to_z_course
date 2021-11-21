# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 00:27:33 2021

@author: ivan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv("Position_Salaries.csv")

#splitting dependend and independend variables
X = dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values

#creating a simple linear regression model - for comparisson
regressor = LinearRegression()
regressor.fit(np.array(X).reshape(-1, 1), np.array(y).reshape(-1, 1))
y_pred = regressor.predict(np.array(X).reshape(-1, 1))

#preprocessing for polinomial linear model
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(np.array(X).reshape(-1, 1))

#creating and training the PLM
regressor_poly = LinearRegression()
regressor_poly.fit(X_poly ,y)
y_poly_pred = regressor_poly.predict(X_poly)

plt.scatter(X,y, color="red")
plt.plot(X,y_pred,color="yellow")
plt.plot(X,y_poly_pred,color="blue")

