import pandas as pd
import mysql.connector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\TANISHA\PROJECTS\Predictive Maintenance (PdM) for Industrial Machinery\pdm_preprocessed.csv")  # use your actual CSV filename
    return df

df = load_data()

# 2. Preprocessing
X = df.drop(columns=['Machine_failure'])  # Drop target column
X = X.select_dtypes(include=['number'])  # Keep only numeric columns


# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = df['Machine_failure']  # Define the target variable


# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. XGBoost Regressor
model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# 5. Predictions and Metrics
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)

# 6. Visualization: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Failure Score')
plt.ylabel('Predicted Failure Score')
plt.title('XGBoost: Actual vs Predicted')
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Feature Importance Plot (optional but recommended!)
plt.figure(figsize=(10, 6))
sns.barplot(x=model.feature_importances_, y=X.columns)
plt.title("Feature Importance - XGBoost")
plt.tight_layout()
plt.show()


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

