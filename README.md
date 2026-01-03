# Telco Customer Churn Prediction
### End-to-End Machine Learning Pipeline & Evaluation

**Goal:** Build a complete ML workflow to predict customer churn using the original IBM Telco dataset, including data validation, feature engineering, model training, evaluation, and actionable business insights.  

**Why this matters:**  
Predicting churn helps telecom companies identify at-risk customers, enabling targeted retention strategies that can improve revenue and customer lifetime value.


## Features

- **Data Pipeline**: Automated data loading, validation, and cleaning (handles missing values, categorical mapping).
- **Feature Engineering**: Selection and standardization of numerical features to prevent data leakage.
- **Model Training**: Comparison of Logistic Regression and Random Forest using 5-fold cross-validation.
- **Evaluation**: Comprehensive metrics including AUC, confusion matrices, classification reports, and feature importance.
- **Business Insights**: Actionable recommendations for retention strategies based on model outputs.
- **Modular Code**: Clean, reusable codebase with separate modules for data, features, training, and evaluation.

## Requirements
- IBM Telco Customer Churn dataset (https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Python 3.8+
- Libraries: pandas, scikit-learn, matplotlib, seaborn

# Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Results

- **Model Performance**: Logistic Regression outperforms Random Forest with higher cross-validated AUC.
- **Key Insights**: Tenure and monthly charges are top predictors of churn.
- **Business Impact**: Enables targeted retention, potentially saving thousands in customer lifetime value.

