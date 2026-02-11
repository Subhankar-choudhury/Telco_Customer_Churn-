# Telco Customer Churn Prediction

**Track 1 – Tabular Machine Learning**

---

## 1. Project Objective

The objective of this project is to predict whether a telecom customer will churn (Yes/No) using the Telco Customer Churn dataset.

This project implements a complete machine learning workflow including:

* Data loading and cleaning
* Feature preprocessing and encoding
* Model training (baseline and improved model)
* Performance evaluation
* Error analysis using confusion matrices

The goal is to build a reproducible and interpretable churn prediction system.

---

## 2. Dataset Description

**Source:** Kaggle – Telco Customer Churn Dataset

The dataset contains 7,043 customer records and 21 features describing:

* Demographics (gender, SeniorCitizen, Partner, Dependents)
* Service subscriptions (InternetService, Contract, PaymentMethod, etc.)
* Billing details (MonthlyCharges, TotalCharges)
* Target variable: **Churn (Yes/No)**

The original dataset was used and cleaned programmatically within the training pipeline.

---

## 3. Data Cleaning & Preprocessing

The following preprocessing steps were performed inside `train.py`:

1. Converted `TotalCharges` from string to numeric format.
2. Removed rows with missing `TotalCharges`.
3. Dropped `customerID` as it is a non-predictive unique identifier.
4. Applied binary encoding to:

   * Partner
   * Dependents
   * PhoneService
   * PaperlessBilling
   * gender
   * Churn (target variable)
5. Applied one-hot encoding to remaining categorical features using `pd.get_dummies()`.
6. Applied `StandardScaler` to features for Logistic Regression.

These steps ensured the dataset was fully numeric and suitable for machine learning models.

---

## 4. Train / Validation Split Method  ✅

The dataset was split using a **stratified 80:20 train-test split**:

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

* 80% used for training
* 20% used for validation/testing
* Stratification preserved churn class distribution
* `random_state=42` ensures reproducibility

This ensures balanced representation of churn and non-churn customers in both sets.

---

## 5. Models Used

### Baseline Model – Logistic Regression

* Trained on scaled features
* Provides a simple and interpretable benchmark
* Suitable for linear decision boundaries

### Improved Model – Gradient Boosting Classifier

* Ensemble boosting algorithm
* Captures non-linear feature relationships
* Iteratively improves weak learners
* More effective for structured tabular datasets

The improved model was introduced to enhance predictive performance.

---

## 6. Metrics Reported  ✅

The following evaluation metrics were used:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Metrics were generated using:

```python
classification_report()
confusion_matrix()
```

---

## 7. Best Result  ✅

The **Gradient Boosting model** achieved the best overall performance.

Key improvements observed:

* Improved recall for churn customers (minority class)
* Better balance between precision and recall
* Reduced classification errors compared to Logistic Regression
* Stronger modeling of non-linear relationships

Since churn prediction prioritizes identifying at-risk customers, improved recall for the churn class makes Gradient Boosting the preferred model.

---

## 8. Error Analysis

Confusion matrices were generated for both models and compared to analyze classification errors.

* Logistic Regression provides a strong baseline but may struggle with complex non-linear patterns.
* Gradient Boosting reduces misclassification errors and improves churn detection.
* False negatives (missed churners) were analyzed carefully due to their potential business impact.

### Confusion Matrix Comparison

![Confusion Matrix Comparison](<img width="1400" height="600" alt="image" src="https://github.com/user-attachments/assets/ca228f2f-5e37-47bf-a820-9b3b05c5450e" />
)

The above visualization compares the performance of both models and highlights the improved model’s ability to better classify churn customers.

---

## 9. Repository Structure

```
train.py               -> Data preprocessing and model training
evaluation.py          -> Model evaluation and confusion matrix visualization
baseline_model.pkl     -> Saved Logistic Regression model
improved_model.pkl     -> Saved Gradient Boosting model
scaler.pkl             -> Saved feature scaler
X_test.csv             -> Saved test features
y_test.csv             -> Saved test labels
error_analysis.png     -> Confusion matrix visualization
```

---

## 10. Conclusion

This project demonstrates that customer churn can be effectively predicted using machine learning techniques applied to structured telecom data.

Logistic Regression provides a strong baseline model, while Gradient Boosting improves performance by modeling complex feature interactions and reducing classification errors.

The workflow is fully reproducible, follows a structured machine learning pipeline, and satisfies all requirements for Track 1 (Tabular ML).

---


