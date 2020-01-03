# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:10:58 2019

@author: İsmailAkdağ & CemGöçen
"""

#Regression Analysis for Antenna Dataset
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, max_error

# Importing the dataset
dataset = pd.read_csv('AntennaDataset.csv')
x = dataset.iloc[:, :6].values
y = dataset.iloc[:, 6:7].values
# Feature Scaling
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.33, random_state=1)
# Fitting the Support Vector Regression Model to the dataset
from sklearn.preprocessing import StandardScaler

scaled_x = StandardScaler()
scaled_y = StandardScaler()
scaled_X = StandardScaler()
scaled_Y = StandardScaler()
x = scaled_x.fit_transform(xtrain)
y = scaled_y.fit_transform(ytrain)
X = scaled_X.fit_transform(xtest)
Y = scaled_Y.fit_transform(ytest)

# Importing the dataset for Multi Output Regressor
dataset_multi = pd.read_csv('AntennaDataset2.csv')
x_multi = dataset_multi.iloc[:, :6].values
y_multi = dataset_multi.iloc[:, 6:8].values
# Feature Scaling
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor


x_multitrain, x_multitest, y_multitrain, y_multitest = train_test_split(x_multi, y_multi, test_size=0.33, random_state=1)
# Fitting the Support Vector Regression Model to the dataset
from sklearn.preprocessing import StandardScaler

scaled_x_multi = StandardScaler()
scaled_y_multi = StandardScaler()
scaled_X_multi = StandardScaler()
scaled_Y_multi = StandardScaler()
x_multi = scaled_x_multi.fit_transform(x_multitrain)
y_multi = scaled_y_multi.fit_transform(y_multitrain)
X_multi = scaled_X_multi.fit_transform(x_multitest)
Y_multi = scaled_Y_multi.fit_transform(y_multitest)

max_depth = 30

# Dataset Visualize

# Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=6)
poly = poly_reg.fit_transform(xtrain)
lin_reg = LinearRegression()
lin_reg.fit(poly, ytrain)
y_pred_lr = lin_reg.predict(poly_reg.fit_transform(xtest))

# Results

print("Results for Polynomial Regression with these features: ", poly_reg)
print("R2 Value:", r2_score(ytest, y_pred_lr))
print("R2 Value: %", r2_score(ytest, y_pred_lr) * 100)

print("Mean Squared Error:", mean_squared_error(ytest, y_pred_lr))
print("Mean Squared Error: %", mean_squared_error(ytest, y_pred_lr) * 100)

print("Root Mean Squared Error:", math.sqrt(mean_squared_error(ytest, y_pred_lr)))
print("Root Mean Squared Error: %", math.sqrt(mean_squared_error(ytest, y_pred_lr)) * 100)

print("Mean Absolute Error:", mean_absolute_error(ytest, y_pred_lr))
print("Mean Absolute Error: %", mean_absolute_error(ytest, y_pred_lr) * 100)

print("Maximum Error:", max_error(ytest, y_pred_lr))
print("Maximum Error: %", max_error(ytest, y_pred_lr) * 100)

print("Total Number of Instances:", len(ytest) + len(y))

print("\n ")

# Gradient Boosting Regressor


from sklearn import ensemble

params = {'n_estimators': 2000, 'max_depth': 8, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
gbr = ensemble.GradientBoostingRegressor(**params)

gbr.fit(xtrain, ytrain)
y_pred_gbr = gbr.predict(xtest)
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

# Results

print("\n ")
print("Results for Gradient Boosting Regressor with these features: ", params)

print("R2 Value:", r2_score(ytest, y_pred_gbr))
print("R2 Value: %", r2_score(ytest, y_pred_gbr) * 100)

print("Mean Squared Error:", mean_squared_error(ytest, y_pred_gbr))
print("Mean Squared Error: %", mean_squared_error(ytest, y_pred_gbr) * 100)

print("Root Mean Squared Error:", math.sqrt(mean_squared_error(ytest, y_pred_gbr)))
print("Root Mean Squared Error: %", math.sqrt(mean_squared_error(ytest, y_pred_gbr)) * 100)

print("Mean Absolute Error:", mean_absolute_error(ytest, y_pred_gbr))
print("Mean Absolute Error: %", mean_absolute_error(ytest, y_pred_gbr) * 100)

print("Maximum Error:", max_error(ytest, y_pred_gbr))
print("Maximum Error: %", max_error(ytest, y_pred_gbr) * 100)

print("Total Number of Instances:", len(ytest) + len(y))

# Random Forrest Regressor

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=10, random_state=0)
rf.fit(xtrain, ytrain)
y_pred_rf = rf.predict(xtest)
# Results

print("\n ")
print("Results for Random Forest Regressor with these features: ", rf)

print("R2 Value:", r2_score(ytest, y_pred_rf))
print("R2 Value: %", r2_score(ytest, y_pred_rf) * 100)

print("Mean Squared Error:", mean_squared_error(ytest, y_pred_rf))
print("Mean Squared Error: %", mean_squared_error(ytest, y_pred_rf) * 100)

print("Root Mean Squared Error:", math.sqrt(mean_squared_error(ytest, y_pred_rf)))
print("Root Mean Squared Error: %", math.sqrt(mean_squared_error(ytest, y_pred_rf)) * 100)

print("Mean Absolute Error:", mean_absolute_error(ytest, y_pred_rf))
print("Mean Absolute Error: %", mean_absolute_error(ytest, y_pred_rf) * 100)

print("Maximum Error:", max_error(ytest, y_pred_rf))
print("Maximum Error: %", max_error(ytest, y_pred_rf) * 100)

print("Total Number of Instances:", len(ytest) + len(y))

# Bayesian Ridge Regressor

from sklearn.linear_model import BayesianRidge

br = BayesianRidge()
br.fit(xtrain, ytrain)
y_pred_br = br.predict(xtest)

# Results

print("\n ")
print("Results for Bayesian Ridge Regressor with these features: ")

print("R2 Value:", r2_score(ytest, y_pred_br))
print("R2 Value: %", r2_score(ytest, y_pred_br) * 100)

print("Mean Squared Error:", mean_squared_error(ytest, y_pred_br))
print("Mean Squared Error: %", mean_squared_error(ytest, y_pred_br) * 100)

print("Root Mean Squared Error:", math.sqrt(mean_squared_error(ytest, y_pred_br)))
print("Root Mean Squared Error: %", math.sqrt(mean_squared_error(ytest, y_pred_br)) * 100)

print("Mean Absolute Error:", mean_absolute_error(ytest, y_pred_br))
print("Mean Absolute Error: %", mean_absolute_error(ytest, y_pred_br) * 100)

print("Maximum Error:", max_error(ytest, y_pred_br))
print("Maximum Error: %", max_error(ytest, y_pred_br) * 100)

print("Total Number of Instances:", len(ytest) + len(y))

# Voting Regressor

from sklearn.ensemble import VotingRegressor

er = VotingRegressor([('lr', lin_reg), ('rf', rf), ('gbr', gbr), ('br', br)])
# er= VotingRegressor([('lr', lin_reg), ('rf', rf)])
er.fit(xtrain, ytrain)
y_pred_vot = er.predict(xtest)

# Results


print("\n ")
print("Results for Voting Regressor with these features: ")

print("R2 Value:", r2_score(ytest, y_pred_vot))
print("R2 Value: %", r2_score(ytest, y_pred_vot) * 100)

print("Mean Squared Error:", mean_squared_error(ytest, y_pred_vot))
print("Mean Squared Error: %", mean_squared_error(ytest, y_pred_vot) * 100)

print("Root Mean Squared Error:", math.sqrt(mean_squared_error(ytest, y_pred_vot)))
print("Root Mean Squared Error: %", math.sqrt(mean_squared_error(ytest, y_pred_vot)) * 100)

print("Mean Absolute Error:", mean_absolute_error(ytest, y_pred_vot))
print("Mean Absolute Error: %", mean_absolute_error(ytest, y_pred_vot) * 100)

print("Maximum Error:", max_error(ytest, y_pred_vot))
print("Maximum Error: %", max_error(ytest, y_pred_vot) * 100)

print("Total Number of Instances:", len(ytest) + len(y))

## MultiOutput Regressor
regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=5000,max_depth=max_depth,random_state=0))

regr_multirf.fit(x_multitrain, y_multitrain)
y_multi_pred_multirf = regr_multirf.predict(x_multitest)

print("\n")
print("Results for Multi Output Regressor: ")

print("R2 Value:", r2_score(y_multitest, y_multi_pred_multirf))
print("R2 Value: %", r2_score(y_multitest, y_multi_pred_multirf) * 100)

print("Mean Squared Error:", mean_squared_error(y_multitest, y_multi_pred_multirf))
print("Mean Squared Error: %", mean_squared_error(y_multitest, y_multi_pred_multirf) * 100)

print("Root Mean Squared Error:", math.sqrt(mean_squared_error(y_multitest, y_multi_pred_multirf)))
print("Root Mean Squared Error: %", math.sqrt(mean_squared_error(y_multitest, y_multi_pred_multirf)) * 100)

print("Mean Absolute Error:", mean_absolute_error(y_multitest, y_multi_pred_multirf))
print("Mean Absolute Error: %", mean_absolute_error(y_multitest, y_multi_pred_multirf) * 100)
print("Total Number of Instances:", len(y_multitest)+len(y_multi))

# Plotting
plt.figure()
plt.plot(y_pred_gbr, 'gd', label='GradientBoostingRegressor')
plt.plot(y_pred_rf, 'b^', label='RandomForestRegressor')
plt.plot(y_pred_lr, 'ys', label='LinearRegression')
plt.plot(y_pred_vot, 'r*', label='VotingRegressor')
plt.tick_params(axis='x', which='both', bottom=False, top=False,
                labelbottom=False)
plt.ylabel('predicted')
plt.xlabel('training samples')
plt.legend(loc="best")
plt.title('Comparison of individual predictions with averaged')
plt.show()

# Comparison

print("Comparison of R2 values: ")

# print("R2 Value for Polynomial Regressor:", r2_score(ytest,y_pred_lr))
print("R2 Value for Polynomial Regressor: %", r2_score(ytest, y_pred_lr) * 100)
# print("R2 Value for Random Forest Regressor:", r2_score(ytest,y_pred_rf))
print("R2 Value for Random Forest Regressor: %", r2_score(ytest, y_pred_rf) * 100)
# print("R2 Value for Gradient Boosting Regressor:", r2_score(ytest,y_pred_gbr))
print("R2 Value for Gradient Boosting Regressor: %", r2_score(ytest, y_pred_gbr) * 100)
# print("R2 Value for Bayesian Ridge Regressor:", r2_score(ytest,y_pred_br))
print("R2 Value for Bayesian Ridge Regressor: %", r2_score(ytest, y_pred_br) * 100)
# print("R2 Value for Voting Regressor:", r2_score(ytest,y_pred_vot))
print("R2 Value for Voting Regressor: %", r2_score(ytest, y_pred_vot) * 100)

#
df = pd.DataFrame({'Actual': ytest.flatten(), 'Predicted': y_pred_lr.flatten()})

df1 = df.head(20)
df1.plot(kind='bar', figsize=(16, 10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.title('Polynomial Regression Actual vs Predicted Values')
plt.show()

df = pd.DataFrame({'Actual': ytest.flatten(), 'Predicted': y_pred_rf.flatten()})

df1 = df.head(20)
df1.plot(kind='bar', figsize=(16, 10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.title('Random Forest Regressor Actual vs Predicted Values')
plt.show()

df = pd.DataFrame({'Actual': ytest.flatten(), 'Predicted': y_pred_gbr.flatten()})

df1 = df.head(20)
df1.plot(kind='bar', figsize=(16, 10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.title('Gradient Boosting Regressor Actual vs Predicted Values')
plt.show()

df = pd.DataFrame({'Actual': ytest.flatten(), 'Predicted': y_pred_br.flatten()})

df1 = df.head(20)
df1.plot(kind='bar', figsize=(16, 10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.title('Bayesian Ridge Regressor Actual vs Predicted Values')
plt.show()

df = pd.DataFrame({'Actual': ytest.flatten(), 'Predicted': y_pred_vot.flatten()})

df1 = df.head(20)
df1.plot(kind='bar', figsize=(16, 10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.title('Voting Regressor Actual vs Predicted Values')
plt.show()

df = pd.DataFrame({'Actual': y_multitest.flatten(), 'Predicted': y_multi_pred_multirf.flatten()})

df1 = df.head(20)
df1.plot(kind='bar', figsize=(16, 10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.title('Multiple Output Regressor w Random Forest Actual vs Predicted Values')
plt.show()

import plotly.graph_objects as go
#####
r2_lr=r2_score(ytest,y_pred_lr)
msqe_lr=mean_squared_error(ytest,y_pred_lr)
rmse_lr=math.sqrt(mean_squared_error(ytest,y_pred_lr))
mae_lr=mean_absolute_error(ytest,y_pred_lr)
maxe_lr=max_error(ytest,y_pred_lr)
#####
r2_rf=r2_score(ytest,y_pred_rf)
msqe_rf=mean_squared_error(ytest,y_pred_rf)
rmse_rf=math.sqrt(mean_squared_error(ytest,y_pred_rf))
mae_rf=mean_absolute_error(ytest,y_pred_rf)
maxe_rf=max_error(ytest,y_pred_rf)
#####
r2_gbr=r2_score(ytest,y_pred_gbr)
msqe_gbr=mean_squared_error(ytest,y_pred_gbr)
rmse_gbr=math.sqrt(mean_squared_error(ytest,y_pred_gbr))
mae_gbr=mean_absolute_error(ytest,y_pred_gbr)
maxe_gbr=max_error(ytest,y_pred_gbr)
#####
r2_br=r2_score(ytest,y_pred_br)
msqe_br=mean_squared_error(ytest,y_pred_br)
rmse_br=math.sqrt(mean_squared_error(ytest,y_pred_br))
mae_br=mean_absolute_error(ytest,y_pred_br)
maxe_br=max_error(ytest,y_pred_br)
#####
r2_vot=r2_score(ytest,y_pred_vot)
msqe_vot=mean_squared_error(ytest,y_pred_vot)
rmse_vot=math.sqrt(mean_squared_error(ytest,y_pred_vot))
mae_vot=mean_absolute_error(ytest,y_pred_vot)
maxe_vot=max_error(ytest,y_pred_vot)
#####
r2_multi=r2_score(y_multitest, y_multi_pred_multirf)
msqe_multi=mean_squared_error(y_multitest, y_multi_pred_multirf)
rmse_multi=math.sqrt(mean_squared_error(y_multitest, y_multi_pred_multirf))
mae_multi=mean_absolute_error(y_multitest, y_multi_pred_multirf)


fig = go.Figure(data=[go.Table(
    header=dict(values=['Results','Polynomial Regressor', 'Random Forest Regressor','Gradient Boosting Regressor','Bayesian Ridge Regressor','Voting Regressor','Multi Output Regressor'],
                line_color='darkslategray',
                fill_color='lightskyblue',
                align='left'),
    cells=dict(values=[['R2 Score %', 'Mean Squared Error %', 'Root Mean Squared Error %', 'Mean Absolute Error %', 'Maximum Error %'],  # 1st column
                       [r2_lr*100,msqe_lr*100,rmse_lr*100,mae_lr*100,maxe_lr*100],
                       [r2_rf*100,msqe_rf*100,rmse_rf*100,mae_rf*100,maxe_rf*100],
                       [r2_gbr*100,msqe_gbr*100,rmse_gbr*100,mae_gbr*100,maxe_gbr*100],
                       [r2_br*100,msqe_br*100,rmse_br*100,mae_br*100,maxe_br*100],
                       [r2_vot*100,msqe_vot*100,rmse_vot*100,mae_vot*100,maxe_vot*100],
                       [r2_multi*100,msqe_multi*100,rmse_multi*100,mae_multi*100,'---']],  # 2nd column
               line_color='darkslategray',
               fill_color='lightcyan',
               align='left'))
])

fig.update_layout(width=2000, height=2000)
fig.show()






