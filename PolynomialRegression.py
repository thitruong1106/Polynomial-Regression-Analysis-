# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 23:48:38 2022

@author: thi kim
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('bloodpressure.csv')
df.head()

### Creating indpendent and dependent variables
X =df.iloc[:,5:6].values     ### Taking just weight columns using iloc
y =df['SYSTOLIC'].values
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
#polynomial regrression model 
#For degrees varying from 1-14
for i in range(1,15):
  poly = PolynomialFeatures(degree =i)
  x_poly = poly.fit_transform(X)
  
  model = LinearRegression()
  model.fit(x_poly, y)
  y_poly_pred = model.predict(x_poly)
  rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
  r2 = r2_score(y,y_poly_pred)
  print(f'RMSE of polynomial regression of degree {i} '+ str(rmse))
  print(f'square root of the RMSE of degree {i} '+ str(r2))

  plt.scatter(X, y)
  # sort the values of x before line plot
  plt.plot(X, y_poly_pred, color='r')
  plt.title(f'Polynomial Regression of degree {i}')
  plt.xlabel('Weights')
  plt.ylabel('SYSTOLIC')
  plt.show()
#10-fold cross validation
from sklearn.model_selection import cross_val_score 
from sklearn.preprocessing import PolynomialFeatures
#Function for creating a polynomial regrerssion model
def create_polynomial_regression_model(degree):
 poly_features = PolynomialFeatures(degree=degree)
 X_poly = poly_features.fit_transform(X)
 poly = LinearRegression()
 return np.mean(cross_val_score(poly, X_poly, y, cv=5))
poly_cv = []
for i in range(1,14):
 poly_cv.append(create_polynomial_regression_model(i))
#plt.scatter(range(1,14),poly_cv)
print("Polynomial cross validation points", poly_cv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
#Spilt the data, into train and test 
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=1)
training_error, validation_error = [],[]
for d in range(1,14):
    #polynomial features for the current degree for the train set
    X_Poly_train = PolynomialFeatures(degree = d).fit_transform(x_train)
    #polynomial features for the test set
    X_poly_test = PolynomialFeatures(degree = d).fit_transform(x_test)
    #linear regression model
    lr = LinearRegression(fit_intercept=False)
    #Fit the model on the train data
    lr.fit(X_Poly_train, y_train)
    #prediction on the  train data
    y_train_pred = lr.predict(X_Poly_train)
    # prediction on the transformed test data
    y_test_pred = lr.predict(X_poly_test)
    #MSE on the train predictions
    training_error.append(mean_squared_error(y_train, y_train_pred))
    #MSE on the validation
    validation_error.append(mean_squared_error(y_test, y_test_pred))
#Calcualting the best degree
lowest_mse = min(validation_error)
print("Lowest_Mse", lowest_mse)
best_degree = validation_error.index(lowest_mse)
# Print the degree of the best model computed above
print("Location of index of best degree is found",best_degree)
#Location = 0, 1st degree is the best 
#The best degree is 1, as it generaste the lowest RMSE when preforming cross
#validation set. 
#Printing the coefficient of polynomial model with degree of 1 
poly_reg = PolynomialFeatures(degree=1, include_bias=False)
#calcualte polynomail features for degree of 1
X_poly = poly_reg.fit_transform(x_train)
#linear regression model
lin2=LinearRegression().fit(X_poly, y_train)
print("Coefficent",lin2.coef_)
# Plot the errors as a function of increasing d value to visualise the training 
# and testing errors
fig, ax = plt.subplots()
# Plot the training error with labels
plt.plot(np.arange(1, 14), training_error, label="Training Error")
# Plot the validation error with labels
plt.plot(np.arange(1, 14), validation_error, label="Validation Error")
# Set the plot labels
plt.xlabel('Degree of Polynomial')
plt.ylabel('Mean Squared Error')
plt.yscale('log')
plt.show();











