# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 21:31:56 2023
RandomForest
Biofixation rate

@author: shabnam
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl


print('RandomForest')

# Load the dataset-------------------------------------------------------------
data = pd.read_excel('dataset-co2biofixation.xlsx')

# Separate features (inputs) and target variable (output)
X = data[['light (lux)', 'gas flow rate (L/min)', 'CO2%', 
          'volume ratio of microalgae to media']]
y = data['CO2 fixation rate (gCO2/L/day)']
'''
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)
'''

# Load the split data from the file
split_data = np.load('split_data.npz')
X_train = split_data['X_train']
X_test = split_data['X_test']
y_train = split_data['y_train']
y_test = split_data['y_test']

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid for grid search------------------------------------
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a Random Forest regressor
rf_regressor = RandomForestRegressor(random_state=42)

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

print ('\nbest params:', grid_search.best_params_ )

# Get the best model from grid search
best_rf_regressor = grid_search.best_estimator_

# Make predictions on the testing data using the best model
y_pred = best_rf_regressor.predict(X_test_scaled)
y_pred_train = best_rf_regressor.predict(X_train_scaled)


# Evaluate the performance of the regressor using mean squared error and R2 Score
mse_train = mean_squared_error(y_train, y_pred_train)
print("\nMSE-train:", mse_train)
R2_score_train = r2_score(y_train, y_pred_train)
print(f"R2_train: {R2_score_train}")

mse = mean_squared_error(y_test, y_pred)
print("\nMSE_test:", mse)
R2_score = r2_score(y_test, y_pred)
print(f"R2_test: {R2_score}\n")


# Plotting the real data vs predicted data-------------------------------------
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 10
fig, axs= plt.subplots(2, figsize=(8, 10))

axs[0].scatter(y_test, y_pred, c='r', label='Test Data')
axs[0].scatter(y_train, y_pred_train, c='b', label='Train Data')
axs[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axs[0].set_aspect('equal', adjustable='box')
axs[0].set_xlabel('Real (gCO2/L/day)')
axs[0].set_ylabel('Predicted (gCO2/L/day)')
axs[0].set_title('RandomForest\nReal vs. Predicted CO2 fixation rate')
axs[0].legend(fontsize=7)

axs[0].annotate(f"MSE_train: {mse_train:.3f}", xy=(0.7, 0.2), 
             xycoords='axes fraction')
axs[0].annotate(f"R2_train: {R2_score_train:.2f}", xy=(0.7, 0.15), 
             xycoords='axes fraction')
axs[0].annotate(f"MSE_test: {mse:.3f}", xy=(0.7, 0.10), 
             xycoords='axes fraction')
axs[0].annotate(f"R2_test: {R2_score:.2f}", xy=(0.7, 0.05), 
             xycoords='axes fraction')


# Feature importance-----------------------------------------------------------
feature_importance = best_rf_regressor.feature_importances_
feature_names = X.columns
color_map = mcolors.LinearSegmentedColormap.from_list('importance', ['lightblue', 'darkblue'])

axs[1].barh(range(len(feature_importance)), feature_importance, align='center', color=color_map(feature_importance))
axs[1].set_yticks(range(len(feature_importance)))
axs[1].set_yticklabels(feature_names)
axs[1].set_xlabel('Feature Importance')


# Predict CO2 fixation rate for new inputs-------------------------------------
new_inputs = np.array([[1000, 0.5, 9, 0.03], [3000, 0.3, 10, 0.05]])
new_inputs_scaled = scaler.transform(new_inputs)
new_outputs =best_rf_regressor.predict(new_inputs_scaled)
print(f"Predicted CO2 fixation rate: {new_outputs}\n")

#end---------------------------------------------------------------------------
