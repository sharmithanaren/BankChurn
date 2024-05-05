**Kaggle Playground Series: Episode 1 - Customer Churn Prediction**
In this competition, the task was to predict whether a customer continues with their account or closes it (churns) based on various features provided in the dataset.

**Overview**
This repository contains my solution to the competition, along with detailed code for data preprocessing, feature engineering, model building, and evaluation. The competition was hosted on Kaggle, and the goal was to achieve the highest Area Under the ROC Curve (AUC-ROC) score.

**Dataset**
The dataset provided for this competition was generated from a deep learning model trained on the Bank Customer Churn Prediction dataset. It includes features such as credit score, geography, gender, age, tenure, balance, and more.

**Approach**
1. Data Preprocessing and Exploration
Loaded and explored the training and test datasets
Checked for missing values and performed necessary imputation
Applied feature engineering techniques to enhance the dataset
2. Model Building
Utilized various machine learning algorithms such as Logistic Regression, XGBoost, LightGBM, CatBoost, and TensorFlow for classification
Implemented pipeline for each model including preprocessing steps such as encoding, scaling, and feature engineering
3. Model Evaluation
Evaluated models using cross-validation with Stratified K-Fold
Calculated AUC-ROC score as the evaluation metric
4. Ensemble Learning
Implemented a Voting Classifier to combine predictions from multiple models
Optimized weights for ensemble using RidgeClassifier
5. Submission
Generated predictions on the test dataset
Created a submission file in the required format for Kaggle

**Dependencies**
Python 3.x
Libraries: NumPy, Pandas, Matplotlib, Seaborn, TensorFlow, Optuna, Scikit-Learn, XGBoost, LightGBM, CatBoost
