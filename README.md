# Credit Card Fraud Detection

![Type of Project](https://img.shields.io/badge/Type%20of%20Project-Machine%20Learning-orange?style=flat)
![Python](https://img.shields.io/badge/Python-3.9-blue.svg?style=flat&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0.2-orange.svg?style=flat&logo=scikit-learn)
![pandas](https://img.shields.io/badge/pandas-1.4.2-red.svg?style=flat&logo=pandas)
![numpy](https://img.shields.io/badge/numpy-1.22.3-lightblue.svg?style=flat&logo=numpy)
![imblearn](https://img.shields.io/badge/imblearn-0.9.1-green.svg?style=flat&logo=imbalanced-learn)

## Project Overview

This project focuses on detecting fraudulent transactions in credit card data using various machine learning algorithms. The goal is to build a model that can effectively classify legitimate and fraudulent transactions, while addressing the issue of imbalanced datasets. Techniques such as SMOTE and Random Under-Sampling are applied to balance the class distribution.

![Credit Card Fraud]([https://www.example-image-url.com/fraud-image.jpeg](https://www.usatoday.com/money/blueprint/images/uploads/2023/09/20041937/GettyImages-1271123246-e1695198030910.jpg?width=700&fit=cover&format=webp))

## Data Description

The dataset consists of anonymized credit card transactions, with the following key columns:

- `Time`: The time elapsed since the first transaction in the dataset, measured in seconds.
- `V1 to V28`: Principal components obtained from PCA, with features anonymized for confidentiality.
- `Amount`: The transaction amount.
- `Class`: The target variable, where 0 represents legitimate transactions and 1 represents fraudulent transactions.

## Data Preprocessing

Before feeding the data into machine learning models, the following preprocessing steps are performed:

- **Data Scaling**: Applied using MinMaxScaler or StandardScaler to normalize the numerical features.
- **Handling Imbalanced Data**: SMOTE and Random Under-Sampling techniques are used to balance the class distribution.
  
## Models

This project implements and tunes three machine learning models for fraud detection:

1. **Random Forest Classifier**:
   - A highly flexible model that performs well with both classification and regression tasks.
   - Hyperparameters are optimized using RandomizedSearchCV.

2. **Logistic Regression**:
   - A linear model suitable for binary classification tasks, tuned with various penalties and solvers.

3. **Neural Network**:
   - A multi-layer perceptron classifier is used to capture non-linear patterns in the data, optimized with different hidden layers and activation functions.

Each model is tuned using cross-validation, and the best parameters are selected to maximize model performance.

## Evaluation Metrics

The models are evaluated using the following metrics, focusing on their ability to handle imbalanced data:

- **F1-Score**: The harmonic mean of precision and recall, especially useful for imbalanced datasets.
- **Precision-Recall AUC (PR AUC)**: A performance metric suited for imbalanced classes.
- **ROC AUC**: Measures the ability of the model to distinguish between classes.

## Final Model Results

Below are the evaluation results for the final models:

- **Random Forest**:
  - **Train F1-Score**: 0.97
  - **Validation F1-Score**: 0.82
  - **Validation ROC AUC**: 0.95
  - **Validation PR AUC**: 0.91

- **Logistic Regression**:
  - **Train F1-Score**: 0.94
  - **Validation F1-Score**: 0.79
  - **Validation ROC AUC**: 0.91
  - **Validation PR AUC**: 0.88

- **Neural Network**:
  - **Train F1-Score**: 0.95
  - **Validation F1-Score**: 0.80
  - **Validation ROC AUC**: 0.93
  - **Validation PR AUC**: 0.89
