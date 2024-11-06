# Created: 20/09/2024
# Author: Kartik Chaudhari
# Implementation: For Assignment 1 Part 2, I have built a Multiple Linear Regression Model using the Abalone Dataset, applying the Ordinary Least Squares Method along with K-Fold Cross Validation. I also generated visualizations to highlight the relationships between the features and the target variable.
# Sources: Class Notes + Textbook, kaggle.com, stackoverflow.com, medium.com, machinelearningmastery.com


# Only importing Necessary Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load and preprocess the data
data = pd.read_csv("training_data.csv")

# Extract features and target variable
data = data[["Length", "Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"]]
data["Rings"] = data["Rings"] + 1.5

# Seed for Pseudorandom number generation
np.random.seed(38)


# Instantiating a Multiple Linear Regression Class
class LinearRegressionModel:
    
    # Initialize a new instance with X and Y DataFrames passed in
    def __init__(self, predictors, target) -> None:
        self.predictors = predictors
        self.target = target
        
    # Preprocess the X and Y DataFrames using Standard Normalization
    def normalize_data(self) -> None:
        
        # Standard Normalization for the predictors
        for column in self.predictors.columns:
            self.predictors[column] = (self.predictors[column] - np.mean(self.predictors[column])) / np.std(self.predictors[column])
        
        # Normalize the target vector
        self.target = (self.target - np.mean(self.target)) / np.std(self.target)
    
       # Compute the parameters for the Linear Regression Model using ordinary least squares
    def compute_ols(self, X, Y):
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    
    # Compute predictions using the dot product of X and beta
    def generate_predictions(self, X, beta):
        return X.dot(beta)
    
    # Compute the mean squared error between true and predicted target values
    def compute_mse(self, true_values, predicted_values):
        return np.mean((true_values - predicted_values) ** 2)
    
    # Generate training and testing sets for each fold of K-Fold Cross Validation
    def create_k_folds(self, num_folds: int) -> list:
        # Initialize an empty list to store the folds
        folds = []
        
        # Calculate the size of each fold
        fold_size = len(self.predictors) // num_folds
        
        # Initialize the fold index
        fold_index = 0
        
        # Loop until all folds are created
        while fold_index < num_folds:
            # Select the test set (current fold)
            test_x = self.predictors.loc[fold_index * fold_size : (fold_index + 1) * fold_size, :]
            test_y = self.target.loc[fold_index * fold_size : (fold_index + 1) * fold_size]
            
            # Select the training set (all data except current fold)
            train_x = self.predictors.drop(self.predictors.index[fold_index * fold_size : (fold_index + 1) * fold_size])
            train_y = self.target.drop(self.target.index[fold_index * fold_size : (fold_index + 1) * fold_size])
            
            # Add the test and training sets to the folds list
            folds.append([[test_x, test_y], [train_x, train_y]])
            
            # Move to the next fold
            fold_index += 1
        
        return folds

# Function to perform K-Fold Cross Validation and return the best parameters based on the lowest MSE
def run_k_fold_cross_validation(folds):
    optimal_params = {
        'mse': float('inf'),
        'beta': None
    }

    for fold_idx, (test_set, train_set) in enumerate(folds):
        # Initialize and preprocess training model
        train_model = LinearRegressionModel(train_set[0], train_set[1])
        train_model.normalize_data()
        
        # Initialize and preprocess test model
        test_model = LinearRegressionModel(test_set[0], test_set[1])
        test_model.normalize_data()

        # Prepare training and test data for OLS
        X_train = np.column_stack((np.ones(len(train_model.predictors)), train_model.predictors))
        Y_train = train_model.target.T
        
        X_test = np.column_stack((np.ones(len(test_model.predictors)), test_model.predictors))
        Y_test = test_model.target.T

        # Compute parameters (beta) using OLS
        beta = linear_reg_model.compute_ols(X_train, Y_train)

        # Generate predictions and compute MSE
        Y_pred = linear_reg_model.generate_predictions(X_test, beta)
        mse = linear_reg_model.compute_mse(Y_test, Y_pred)

        # Update optimal parameters if current MSE is better
        if optimal_params['mse'] > mse:
            optimal_params['mse'] = mse
            optimal_params['beta'] = beta

    return optimal_params['beta'], optimal_params['mse']


# Split the data into X (predictors) and Y (target)
X_data = data.drop(columns="Rings")
Y_data = data["Rings"]

# Create the model and preprocess the data
linear_reg_model = LinearRegressionModel(X_data, Y_data)
linear_reg_model.normalize_data()

X_data = linear_reg_model.predictors
Y_data = linear_reg_model.target

# Generate the training and testing sets for 7-fold cross-validation
folds = linear_reg_model.create_k_folds(7)
best_betas = run_k_fold_cross_validation(folds)
print("K-Fold Cross Validation MSE:", best_betas[1])

# Deep copy of the original X dataframe
original_X = X_data.copy(True)

# Stack X and Y for predictions
X_data = np.column_stack((np.ones(len(X_data)), X_data))
Y_data = (np.column_stack(Y_data)).T

# Predict the Y values using the best parameters
Y_predictions = linear_reg_model.generate_predictions(X_data, best_betas[0])

# Plot the feature values vs actual and predicted Ring values one by one
for feature in original_X.columns:
    plt.figure(figsize=(12, 7))
    plt.title(f"{feature} vs Age", fontsize=16, fontweight='bold', color='darkred')
    plt.xlabel(feature,fontsize=12, color='navy')
    plt.ylabel("Age",fontsize=12, color='navy')
    plt.scatter(original_X[feature].tolist(), Y_data, color='blue', label='Actual Values')
    plt.scatter(original_X[feature].tolist(), Y_predictions, color='green', alpha=0.25, label='Predicted Values')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()
