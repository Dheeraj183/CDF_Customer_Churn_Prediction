# Customer Churn Prediction

## Objective

Predict whether a customer is likely to churn using machine learning models.

---

## Dataset

Telco Customer Churn Dataset

Source:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

## Exploratory Data Analysis (EDA)

Key findings:

- Approximately 27% of customers churned.
- Customers with month-to-month contracts showed higher churn rates.
- Customers with higher monthly charges were more likely to churn.
- Customers with longer tenure were less likely to churn.

---

## Data Preprocessing

- Removed `customerID` column.
- Converted `TotalCharges` to numeric format.
- Filled missing values using the median.
- Encoded categorical variables using `LabelEncoder`.
- Standardized features using `StandardScaler`.

---

## Models Trained

### Logistic Regression

Accuracy: XX.XX

Precision: XX.XX

Recall: XX.XX

F1-score: XX.XX

### Random Forest

Accuracy: XX.XX

Precision: XX.XX

Recall: XX.XX

F1-score: XX.XX

---

## Best Model

Logistic Regression was selected as the best model because it achieved higher Recall and F1-score. In churn prediction, identifying customers likely to leave is more important than maximizing overall accuracy.

---

## Project Structure

```text
CDF_Customer_churn_Prediction/
│
├── models/
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── feature_columns.pkl
│
├── Customer_Churn_Prediction.ipynb
├── predict_churn.py
├── requirements.txt
├── README.md
└── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## Running Predictions

Run:

```bash
python predict_churn.py
```

Example Output:

```text
Customer is likely to churn.
```

---

## Requirements

```bash
pip install -r requirements.txt
```