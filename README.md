# Breast Cancer Data Analysis and Streamlit App

![Breast Cancer Detection](https://img.shields.io/badge/Breast_Cancer_Detection-ML-brightgreen)

## Project Overview

This project involves the analysis of the Breast Cancer dataset to build a predictive model using an Artificial Neural Network (ANN). The project encompasses data preprocessing, feature selection, model tuning with Grid Search Cross-Validation, and deploying an interactive web application using Streamlit. By completing this project, you will gain hands-on experience in machine learning workflows, model optimization, and web app development.

## Features

- *Data Preprocessing*: Loading and cleaning the dataset to ensure it's ready for analysis.
- *Feature Selection*: Identifying the most significant features using SelectKBest.
- *Model Training*: Building and training an ANN model (MLPClassifier) with hyperparameter tuning.
- *Evaluation*: Assessing model performance using confusion matrix and classification report.
- *Interactive Web App*: Deploying a Streamlit app for user interaction and model predictions.
- *Version Control*: Managing the project with Git and GitHub for collaboration and version tracking.

### Dataset Acquisition and Preparation

1. *Download the Dataset*

   The Breast Cancer dataset can be obtained from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) or directly via sklearn.

2. *Data Preparation*
 checks for missing values, and preprocesses the data for analysis.

### Feature Selection

   Utilizes SelectKBest from sklearn.feature_selection to identify the most significant features. Selected features and the scaler are saved for later use.


### Streamlit App

1. *Run the Streamlit App Locally*

   bash
   streamlit run app.py
   

2. *Interact with the App*

   The Streamlit app allows users to:
   - Input feature values.
   - View model predictions.
   - Explore the dataset interactively.

## Results

### Missing Values

All features in the dataset have been checked for missing values, and none were found.

### Selected Features

The following features were selected for model training:

- mean radius
- mean perimeter
- mean area
- mean concavity
- mean concave points
- worst radius
- worst perimeter
- worst area
- worst concavity
- worst concave points

Selected features are saved to selected_features.pkl, and the scaler is saved to scaler.pkl.

### Model Performance

- *Best Parameters Found:*
  python
  {
      'activation': 'relu',
      'alpha': 0.0001,
      'hidden_layer_sizes': (50,),
      'learning_rate': 'constant',
      'solver': 'adam'
  }
  
- *Best Cross-Validation Accuracy:* 96.04%

- *Confusion Matrix:*
  
  [[42  1]
   [ 2 69]]
  

- *Classification Report:*
  
                precision    recall  f1-score   support

            0       0.95      0.98      0.97        43
            1       0.99      0.97      0.98        71

    accuracy                           0.97       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114
  

The trained model is saved to mlp_model.pkl.
