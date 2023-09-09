# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 20:18:55 2023
ANN
Biofixation rate
GridSearch
@author: shabnam
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import matplotlib.colors as mcolors
import matplotlib as mpl

print('ANN')

# Load the dataset-------------------------------------------------------------
data = pd.read_excel('dataset-co2biofixation.xlsx')

# Separate features (inputs) and target variable (output)
X = data[['light (lux)', 'gas flow rate (L/min)', 'CO2%', 
          'volume ratio of microalgae to media']]
y = data['CO2 fixation rate (gCO2/L/day)']

'''
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3
                                                    , random_state=42)

# Save the split data into separate files
np.savez('split_data.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
'''

# Load the split data from the file
split_data = np.load('split_data.npz')
X_train = split_data['X_train']
X_test = split_data['X_test']
y_train = split_data['y_train']
y_test = split_data['y_test']

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid for grid search------------------------------------
param_grid = {
    'hidden_layer_sizes': [ (n,) for n in range(2, 21)] + 
                          [(n, m) for n in range(2, 21) for m in range(2, 21)],
    #'activation': ['relu', 'tanh'],
    #'solver': ['adam', 'lbfgs'],
    #'alpha': [ 0.01, 0.001, 0.0001],
    #'learning_rate': ['constant', 'adaptive'],
    #'learning_rate_init': [ 0.01, 0.1, 0.2],
    'max_iter': [3000]
}

# Create the MLPRegressor model
model = MLPRegressor( )#random_state=42)

# Perform grid search with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Access the best model
best_model = grid_search.best_estimator_


# Print the best parameters and best score
print("\nBest Parameters:", grid_search.best_params_)
print("Best Score (Negative Mean Squared Error):", grid_search.best_score_)
    

# Fit the best model using the training data
best_model.fit(X_train_scaled, y_train)

# Make predictions on the testing set
y_pred = best_model.predict(X_test_scaled)
y_pred_train = best_model.predict(X_train_scaled)


# Evaluate the model
mse_train = mean_squared_error(y_train, y_pred_train)
print("\nMSE-train:", mse_train)
R2_score_train = r2_score(y_train, y_pred_train)
print(f"R2_train: {R2_score_train}")

mse = mean_squared_error(y_test, y_pred)
print(f"\nMSE_test: {mse}")
R2_score = r2_score(y_test, y_pred)
print(f"R2_test: {R2_score}")

# Plotting the real data vs predicted data-------------------------------------
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 10
fig, axs= plt.subplots(2, figsize=(8, 10))

axs[0].scatter(y_test, y_pred, c='r', label='Test Data')
axs[0].scatter(y_train, y_pred_train, c='b', label='Train Data')
axs[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
axs[0].set_aspect('equal', adjustable='box')
axs[0].set_xlabel('Real (gCO2/L/day)')
axs[0].set_ylabel('Predicted (gCO2/L/day)')
axs[0].set_title('ANN\nReal vs. Predicted CO2 fixation rate')
axs[0].legend(fontsize=7)

axs[0].annotate(f"MSE_train: {mse_train:.3f}", xy=(0.65, 0.2), 
             xycoords='axes fraction')
axs[0].annotate(f"R2_train: {R2_score_train:.2f}", xy=(0.65, 0.15), 
             xycoords='axes fraction')
axs[0].annotate(f"MSE_test: {mse:.3f}", xy=(0.65, 0.10), 
             xycoords='axes fraction')
axs[0].annotate(f"R2_test: {R2_score:.2f}", xy=(0.65, 0.05), 
             xycoords='axes fraction')


# Calculate feature importance using permutation importance--------------------
result = permutation_importance(best_model, X_train, y_train, n_repeats=10)#, random_state=42)
importance = result.importances_mean
features = X.columns
color_map = mcolors.LinearSegmentedColormap.from_list('importance', ['lightblue', 'darkblue'])


axs[1].barh(range(len(importance)), importance, align='center', color=color_map(importance))
axs[1].set_yticks(range(len(importance)))
axs[1].set_yticklabels(features)
axs[1].set_xlabel("Feature Permutation Importance")



# Predict CO2 fixation rate for new inputs-------------------------------------
new_inputs = np.array([[1000, 0.5, 9, 0.03], [3000, 0.3, 10, 0.05]])
new_inputs_scaled = scaler.transform(new_inputs)
new_outputs = best_model.predict(new_inputs_scaled)
print(f"\nPredicted CO2 fixation rate: {new_outputs}")

#end---------------------------------------------------------------------------