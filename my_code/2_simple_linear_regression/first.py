import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("Salary_Data.csv")

#data preprocessing
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#creating the LR model
regressor = LinearRegression()

#training/fitting the model
regressor.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))

#predicting the test results
y_pred = regressor.predict(X_test.reshape(-1, 1))


#plotting the test dataset, and the predicted valus, acquire from our linear regression model
pt.scatter(X_test, y_test, color="red")
pt.plot(X_test, y_pred, color="blue")