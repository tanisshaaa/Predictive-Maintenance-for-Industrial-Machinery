import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\TANISHA\PROJECTS\Predictive Maintenance (PdM) for Industrial Machinery\pdm_preprocessed.csv")  # use your actual CSV filename
    return df

df = load_data()

# Step 3: Set the target column
df['FailureWithin24hrs'] = df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].sum(axis=1)
df['FailureWithin24hrs'] = df['FailureWithin24hrs'].apply(lambda x: 1 if x > 0 else 0)

# Step 4: Drop unnecessary columns
X = df.drop(columns=['UDI', 'Product_ID', 'Type'])
y = df['FailureWithin24hrs']

# Step 5: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Step 6: Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Predict and Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

# Step 8: Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 9: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Step 10: Feature Importances
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 6))
sns.barplot(x=importances[indices], y=features[indices])
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# Step 11: Save the model
joblib.dump(model, "rf_failure_predictor.pkl")

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Convert predicted probabilities to binary class if needed
y_pred_class = [1 if prob > 0.5 else 0 for prob in y_pred]

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred_class))

# Precision, Recall, F1
print("Classification Report:\n", classification_report(y_test, y_pred_class))

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_class))

# ROC-AUC Score
# If you used predict_proba(), use [:, 1] for class 1 probability
# Otherwise, use the continuous output directly (for models like XGBoost)
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))

