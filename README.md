# Customer Churn Prediction

## Objective

The objective of this project is to predict whether a customer is likely to churn (leave the company) using machine learning techniques. Identifying customers at risk of churning allows businesses to take proactive retention measures and improve customer satisfaction.

---

## Dataset

**Dataset:** Telco Customer Churn Dataset

**Source:**  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

**Dataset Size:**
- Rows: 7,043
- Columns: 21

**Target Variable:**
- Churn
  - Yes = Customer churned
  - No = Customer retained

---

## Exploratory Data Analysis (EDA)

The following analyses were performed to understand customer behavior and factors influencing churn:

### Key Findings

- Approximately **27% of customers churned**, while **73% remained with the company**.
- Customers with **month-to-month contracts** showed significantly higher churn rates than customers with one-year or two-year contracts.
- Customers with **higher monthly charges** were more likely to churn.
- Customers with **longer tenure** were less likely to churn.
- The dataset showed a moderate class imbalance, making Recall and F1-score important evaluation metrics.

---

## Data Preprocessing

The following preprocessing steps were performed before model training:

1. Removed the `customerID` column because it is only an identifier and does not contribute to prediction.
2. Converted the `TotalCharges` column from object type to numeric.
3. Handled missing values by replacing missing `TotalCharges` values with the median.
4. Encoded categorical features using `LabelEncoder`.
5. Split the data into training and testing sets using an 80-20 ratio.
6. Applied `StandardScaler` for feature scaling.
7. Used `class_weight='balanced'` to address moderate class imbalance.

---

## Models Trained

Two classification models were trained and evaluated:

### 1. Logistic Regression

| Metric | Value |
|----------|----------|
| Accuracy | 0.7395 |
| Precision | 0.5059 |
| Recall | 0.7968 |
| F1-score | 0.6189 |

### 2. Random Forest Classifier

| Metric | Value |
|----------|----------|
| Accuracy | 0.7885 |
| Precision | 0.6338 |
| Recall | 0.4813 |
| F1-score | 0.5471 |

---

## Model Comparison

| Model | Accuracy | Precision | Recall | F1-score |
|---------|---------|---------|---------|---------|
| Logistic Regression | 0.7395 | 0.5059 | 0.7968 | 0.6189 |
| Random Forest | 0.7885 | 0.6338 | 0.4813 | 0.5471 |

---

## Best Model

**Selected Model:** Logistic Regression

### Reason for Selection

Although Random Forest achieved higher accuracy, Logistic Regression achieved significantly higher Recall and F1-score.

For churn prediction, identifying customers who are likely to leave is more important than maximizing overall accuracy. Therefore, Logistic Regression was selected as the final model because it captures a larger proportion of churning customers.

---

## Saved Model Files

The following files are stored in the `models/` directory:

```text
models/
│
├── churn_model.pkl
├── scaler.pkl
├── label_encoders.pkl
└── feature_columns.pkl
```

These files are used by the prediction script to preprocess new data and generate churn predictions.

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
├── customer_churn_prediction.ipynb
├── predict_churn.py
├── requirements.txt
├── README.md
└── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## Running Predictions

The prediction script loads the trained model and predicts churn for new customer data.

### Run Command

```bash
python predict_churn.py
```

### Example Output

```text
Customer is likely to churn.
```

or

```text
Customer is likely to stay.
```

---

## Example Prediction Workflow

1. Load saved model (`churn_model.pkl`)
2. Load scaler and encoders
3. Accept new customer data
4. Apply the same preprocessing used during training
5. Generate churn prediction
6. Display prediction result

---

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

### requirements.txt

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
```

---

## Conclusion

This project demonstrates an end-to-end machine learning workflow for customer churn prediction, including data preprocessing, exploratory data analysis, model training, evaluation, model selection, and deployment-ready prediction scripts. The final Logistic Regression model provides an effective solution for identifying customers at risk of churning and can support proactive customer retention strategies.
