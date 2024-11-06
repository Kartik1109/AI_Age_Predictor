# Abalone Age Prediction Using Regression Models

This project focuses on predicting the age of abalone using linear and polynomial regression models. By analyzing various physical measurements of abalone, the model aims to estimate their age, which traditionally requires a time-consuming and error-prone manual process.

## Table of Contents

- [Context](#context)
- [Challenge](#challenge)
- [Dataset](#dataset)
- [Goal](#goal)
- [Guidelines](#guidelines)
- [Implementation](#implementation)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)

---

## Context

Abalone are marine snails found in various aquatic environments, including the sheltered bays and coastal regions of North America. In Canada, particularly British Columbia and Alaska, the Northern Abalone is a dominant species valued for its meat. Due to its high market price (CAD 28 to CAD 45 per kilogram) and demand, there has been significant illegal harvesting.

Determining the age of an abalone is crucial for sustainable harvesting practices. Currently, the age is determined by cutting through the shell and counting the rings under a microscope—a process that is time-consuming and prone to errors.

## Challenge

The objective is to build a machine learning (ML) model that can predict the age of abalone using measurable physical features, thus reducing the need for manual ring counting. This project involves analyzing the relationship between the abalone's physical characteristics and their age, and developing a regression model for accurate age prediction.

## Dataset

The dataset used for this project is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/abalone). Modifications to the original dataset include:

1. **Sex Column Removed**: The feature representing the sex of the abalone has been dropped.
2. **Data Division**: The dataset has been divided into five subsets:
   - **Training and Testing Sets**: 2,577 samples provided for model development.
   - **Held-Out Test Sets**: Three subsets of 500 samples each (one with 577 samples) are held out for final evaluation.

The dataset features include:

- **Features**:
  - Length
  - Diameter
  - Height
  - Whole Weight
  - Shucked Weight
  - Viscera Weight
  - Shell Weight
- **Target Variable**:
  - Rings (number of rings, which correlates to age)

*Note*: Age is calculated by adding 1.5 to the number of rings.

## Goal

- **Primary Objective**: Build a linear or polynomial regression model to predict the age of abalone using the provided features.
- **Performance Evaluation**: Your model will be evaluated on one of the held-out test sets.
- **Deliverables**:
  - Report the learned θ (theta) values.
  - Visualize and analyze the results.

## Guidelines

To achieve the project's goal, follow these steps:

1. **Data Visualization**: Analyze and visualize all features to understand their relationship with the abalone's age.
2. **Model Selection**: Determine whether a linear or polynomial regression model is more suitable based on the data analysis.
3. **Cost Function Selection**: Choose an appropriate cost function (e.g., RMSE, MSE, MAE) for evaluating the model.
4. **Approach Selection**: Decide between using Gradient Descent (GD) or Ordinary Least Squares (OLS) for training.
5. **Data Splitting**: Select a subset from the provided 2,577 samples to train your model.
6. **Model Training**: Train your regression model using the selected data and approach.
7. **Performance Evaluation**: Assess your model's performance using appropriate metrics.
8. **Parameter Reporting**: Report the learned θ values.
9. **Visualization**: Visualize the predicted values versus the actual values to assess model accuracy.

## Implementation

### 1. Data Visualization

- **Correlation Analysis**: Compute the correlation coefficients between each feature and the target variable.
- **Scatter Plots**: Create scatter plots for each feature against the number of rings to visualize relationships.
- **Histograms**: Plot histograms to understand the distribution of each feature.

### 2. Model Selection

- **Linear Regression**: Start with a linear regression model to establish a baseline.
- **Polynomial Regression**: If non-linear relationships are observed, consider polynomial regression (e.g., degree 2 or 3).

### 3. Cost Function Selection

- **Mean Squared Error (MSE)**: Commonly used for regression tasks to penalize larger errors.
- **Root Mean Squared Error (RMSE)**: Provides error in the same units as the target variable.
- **Mean Absolute Error (MAE)**: Measures the average magnitude of errors.

### 4. Approach Selection

- **Gradient Descent (GD)**: Iteratively updates parameters to minimize the cost function.
- **Ordinary Least Squares (OLS)**: Calculates the parameters analytically for linear regression.

### 5. Data Splitting

- **Training Set**: Use a significant portion (e.g., 70%) of the 2,577 samples for training.
- **Validation Set**: Use the remaining samples to validate and tune the model.

### 6. Model Training

- **Feature Scaling**: Normalize or standardize features to improve model performance.
- **Training Process**:
  - For GD: Initialize parameters and iteratively update them.
  - For OLS: Compute parameters analytically.
- **Hyperparameter Tuning**: If using GD, experiment with different learning rates and epochs.

### 7. Performance Evaluation

- **Compute Cost Function**: Evaluate the model using the selected cost function on the validation set.
- **Cross-Validation**: Optionally perform k-fold cross-validation for more robust evaluation.

### 8. Parameter Reporting

- **θ Values**: Print out the learned θ values (coefficients) for the model.

### 9. Visualization

- **Predicted vs. Actual Plot**: Plot predicted ages against actual ages to assess accuracy.
- **Residual Plot**: Plot residuals to check for patterns that might indicate model issues.

## Results

- **Model Performance**: Report the final cost (error) on the validation set.
- **θ Values**: Present the learned parameters of the model.
- **Visualization**: Include plots showing the fit of the model and the distribution of errors.
- **Interpretation**: Discuss the implications of the results and any patterns observed.

## Requirements

To run the code, you will need:

- **Python 3.x**
- **Libraries**:
  - `numpy` for numerical computations
  - `pandas` for data manipulation
  - `matplotlib` and `seaborn` for plotting
  - `scikit-learn` for machine learning tools

Install dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
