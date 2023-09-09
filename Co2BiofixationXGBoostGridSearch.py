# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 17:01:27 2023

XGBoost CO2 Biofixation rate
GridSearch

@author: shahh
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib as mpl


print ('XGboost')

# Load the dataset-------------------------------------------------------------
data = pd.read_excel('dataset-co2biofixation.xlsx')

# Separate features (inputs) and target variable (output)
X = data[['light (lux)', 'gas flow rate (L/min)', 'CO2%', 
          'volume ratio of microalgae to media']]
y = data['CO2 fixation rate (gCO2/L/day)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)#, random_state=42)


# Save the split data into separate files
np.savez('split_data.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
'''
# Load the split data from the file
split_data = np.load('split_data.npz')
X_train = split_data['X_train']
X_test = split_data['X_test']
y_train = split_data['y_train']
y_test = split_data['y_test']
'''
# Define the parameter grid for grid search------------------------------------
param_grid = {
    'learning_rate': [0.1, 0.2, 0.3],
    'max_depth': [None, 5, 10],
    'n_estimators': [5, 10, 15, 20]
}

# Create the XGBoost regression model
model = xgb.XGBRegressor()

# Create a GridSearchCV object
grid_search= GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
print('\nbest parameters', best_params)

# Make predictions on the testing set
y_pred = best_model.predict(X_test)

y_pred_train = best_model.predict(X_train)

# Evaluate the model
mse_train = mean_squared_error(y_train, y_pred_train)
print("\nMSE-train:", mse_train)
R2_score_train = r2_score(y_train, y_pred_train)
print(f"R2_Score_train: {R2_score_train}")

mse = mean_squared_error(y_test, y_pred)
print(f"\nMSE: {mse}")
R2_score = r2_score(y_test, y_pred)
print(f"R2_Score: {R2_score}")


#Plots-------------------------------------------------------------------------
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 10
fig, axs = plt.subplots(2, figsize=(8, 10))

# Scatter plot of true values vs. predicted values
axs[0].scatter(y_test, y_pred, c='r', label='Test Data')
axs[0].scatter(y_train, y_pred_train, c='b', label='Train Data')
axs[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axs[0].set_aspect('equal', adjustable='box')
axs[0].set_xlabel('Real (gCO2/L/day)')
axs[0].set_ylabel('Predicted (gCO2/L/day)')
axs[0].set_title('XGBoost\nReal vs. Predicted CO2 fixation rate')
axs[0].legend(fontsize=7)

axs[0].annotate(f'MSE_train: {mse_train:.3f}', xy=(0.7, 0.20), 
                xycoords='axes fraction')
axs[0].annotate(f'R2_train: {R2_score_train:.2f}', xy=(0.7, 0.15), 
                xycoords='axes fraction')
axs[0].annotate(f'MSE_test: {mse:.3f}', xy=(0.7, 0.10), 
                xycoords='axes fraction')
axs[0].annotate(f'R2_test: {R2_score:.2f}', xy=(0.7, 0.05), 
                xycoords='axes fraction')

# Plot feature importances
importance = best_model.feature_importances_
feature_names = X.columns
color_map = mcolors.LinearSegmentedColormap.from_list('importance', ['lightblue', 'darkblue'])

axs[1].barh(range(len(importance)), importance, align='center', color=color_map(importance))
axs[1].set_yticks(range(len(importance)))
axs[1].set_yticklabels(feature_names)
axs[1].set_xlabel('Feature Importance')


#Predict CO2 fixation rate for new inputs--------------------------------------
new_inputs = np.array([[1000, 0.5, 9, 0.03], [3000, 0.3, 10, 0.05]])
new_outputs = best_model.predict(new_inputs)
print(f"\nPredicted CO2 fixation rate: {new_outputs}")
