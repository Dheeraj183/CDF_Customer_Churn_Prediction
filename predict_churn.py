#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import pandas as pd

# Load saved files
with open("models/churn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("models/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Example new customer data
new_customer = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 1026.0
}])

# Encode categorical columns
for col, encoder in label_encoders.items():
    new_customer[col] = encoder.transform(new_customer[col])

# Keep same column order
new_customer = new_customer[feature_columns]

# Scale data
new_customer_scaled = scaler.transform(new_customer)

# Predict
prediction = model.predict(new_customer_scaled)

if prediction[0] == 1:
    print("Customer is likely to churn.")
else:
    print("Customer is likely to stay.")


# In[ ]:





# In[ ]:




