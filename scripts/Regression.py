import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    df = pd.read_csv(pdm_preprocessed.csv")  # use your actual CSV filename
    return df

df = load_data()

# Create regression target (modify this logic as per real failure score)
df['FailureScore'] = df['Air_temperature_K'] * 0.2 + df['Rotational_speed_rpm'] * 0.01

# Define features and target
X = df.drop(columns=['FailureScore', 'UDI', 'Product_ID', 'Type'], errors='ignore')
y = df['FailureScore']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate regression performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Regression Performance:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")

# Visualization
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Failure Score")
plt.ylabel("Predicted Failure Score")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------
# OPTIONAL: Classification interpretation (binary thresholding)
# --------------------------------------
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Binary conversion (example threshold - you can adjust based on domain knowledge)
threshold = y.median()  # using median as threshold for classification
y_pred_class = [1 if pred > threshold else 0 for pred in y_pred]
y_test_class = [1 if actual > threshold else 0 for actual in y_test]

print("\n📊 Classification Metrics (using threshold = median):")

# Check if both classes are present
if len(set(y_test_class)) < 2:
    print("⚠️ Only one class present in y_test. Classification metrics like ROC-AUC are not defined.")
else:
    print("Accuracy:", accuracy_score(y_test_class, y_pred_class))
    print("Classification Report:\n", classification_report(y_test_class, y_pred_class))
    print("Confusion Matrix:\n", confusion_matrix(y_test_class, y_pred_class))
    print("ROC-AUC Score:", roc_auc_score(y_test_class, y_pred))  # use raw y_pred for AUC
