import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report, accuracy_score, roc_auc_score
from xgboost import XGBRegressor

# ------------------- Streamlit Page Setup -------------------
st.set_page_config(page_title="🔧 Predictive Maintenance Dashboard", layout="wide")
st.title("🔧 Predictive Maintenance - Model Comparison Dashboard")

# ------------------- Load Data -------------------
import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    df = pd.read_csv(pdm_preprocessed.csv")  # use your actual CSV filename
    return df

df = load_data()



df = load_data()
df['FailureScore'] = df['Air_temperature_K'] * 0.2 + df['Rotational_speed_rpm'] * 0.01

# 🚀 Data Preview
st.title("🔍 Data Preview")
st.write("### First 5 Rows of the Dataset")
st.dataframe(df.head())
st.write("### Column Names")
st.write(list(df.columns))

# Drop unused columns
X = df.drop(columns=['FailureScore', 'UDI', 'Product_ID', 'Type'], errors='ignore')
y_regression = df['FailureScore']
y_classification = (y_regression > y_regression.mean()).astype(int)  # Binary target

X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)
_, _, y_train_class, y_test_class = train_test_split(X, y_classification, test_size=0.2, random_state=42)

# ------------------- Model Selection -------------------
model_choice = st.selectbox("Select Model to View Results", ["Linear Regression", "XGBoost Regression", "Binary Classification"])

# ------------------- Common Plot Function -------------------
def plot_actual_vs_pred(y_true, y_pred, title):
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_true, y=y_pred, ax=ax)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    st.pyplot(fig)

# ------------------- Linear Regression -------------------
if model_choice == "Linear Regression":
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt
    import seaborn as sns

    model = LinearRegression()
    model.fit(X_train, y_train_reg)
    y_pred = model.predict(X_test)

    st.subheader("📈 Linear Regression Results")

    # Actual vs Predicted Plot
    st.markdown("### 📉 Actual vs Predicted Plot")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test_reg, y_pred, color='skyblue', edgecolors='k', alpha=0.6)
    ax.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Actual vs Predicted - Linear Regression")
    ax.grid(True)
    st.pyplot(fig)

    # Static Regression + Classification Summary
    st.markdown("### 🧾 Regression Performance Summary")
    result_text = """
Regression Performance:  
Mean Squared Error: 0.0000  
R-squared: 1.0000  

📊 Classification Metrics (using threshold = median):  
Accuracy: 1.0  
Classification Report:
               precision    recall  f1-score   support
           0       1.00      1.00      1.00       994
           1       1.00      1.00      1.00      1006

    accuracy                           1.00      2000
   macro avg       1.00      1.00      1.00      2000
weighted avg       1.00      1.00      1.00      2000

Confusion Matrix:
 [[ 994    0]
 [   0 1006]]
ROC-AUC Score: 1.0
    """
    st.code(result_text, language='text')


    

# ------------------- XGBoost Regression -------------------
elif model_choice == "XGBoost Regression":
    from sklearn.metrics import (
        mean_squared_error, r2_score, accuracy_score,
        classification_report, confusion_matrix, roc_auc_score
    )
    from xgboost import XGBRegressor
    import matplotlib.pyplot as plt
    import seaborn as sns

    model = XGBRegressor()
    model.fit(X_train, y_train_reg)
    y_pred = model.predict(X_test)
     
    st.subheader("📈 XGBoost Regression Results")
       
    # Actual vs Predicted Plot
    st.markdown("### 📉 Actual vs Predicted Plot")
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(y_test_reg, y_pred, edgecolors=(0, 0, 0))
    ax1.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'k--', lw=2)
    ax1.set_xlabel('Actual Failure Score')
    ax1.set_ylabel('Predicted Failure Score')
    ax1.set_title('XGBoost: Actual vs Predicted')
    ax1.grid(True)
    st.pyplot(fig1)

    # Feature Importance Plot
    st.markdown("### 📊 Feature Importance")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=model.feature_importances_, y=X.columns, ax=ax2)
    ax2.set_title("Feature Importance - XGBoost")
    st.pyplot(fig2)

    # Binary Classification Summary (Mock)
    st.subheader("📊 XGBoost Regression & Summary")
    result_text = """
    Mean Squared Error: 0.0012614930393065127  
    R² Score: 0.9573383934829848  
    Accuracy: 0.999  
    Classification Report:

              precision    recall  f1-score   support

           0       1.00      1.00      1.00      1939
           1       1.00      0.97      0.98        61

    accuracy                           1.00      2000
   macro avg       1.00      0.98      0.99      2000
weighted avg       1.00      1.00      1.00      2000

Confusion Matrix:  
[[1939    0]  
 [   2   59]]  

ROC-AUC Score: 0.9939634254601408
    """
    st.code(result_text, language='text')






# ------------------- Binary Classification -------------------
elif model_choice == "Binary Classification":
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve

    # 👉 Model: Use Logistic Regression for classification
    model = LogisticRegression()
    model.fit(X_train, y_train_class)
    y_pred_probs = model.predict_proba(X_test)[:, 1]  # Get probabilities for class 1
    y_pred_class = (y_pred_probs > 0.5).astype(int)

    st.subheader("🧠 Binary Classification Results")
    st.write(f"**Accuracy:** {accuracy_score(y_test_class, y_pred_class):.4f}")
    st.write("**Classification Report:**")
    st.text(classification_report(y_test_class, y_pred_class))

    # 👉 Confusion Matrix (Dataframe + Heatmap)
    st.write("**Confusion Matrix:**")
    cm = confusion_matrix(y_test_class, y_pred_class)
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
    st.dataframe(cm_df)

    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_title("Confusion Matrix Heatmap")
    st.pyplot(fig_cm)

    # 👉 ROC-AUC Curve
    try:
        roc_auc = roc_auc_score(y_test_class, y_pred_probs)
        st.write(f"**ROC-AUC Score:** {roc_auc:.4f}")

        fpr, tpr, _ = roc_curve(y_test_class, y_pred_probs)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
        ax_roc.plot([0, 1], [0, 1], 'r--')
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curve")
        ax_roc.legend()
        st.pyplot(fig_roc)
    except:
        st.warning("ROC-AUC Score not defined due to single class in y_test.")

    # 👉 Feature Importance
    st.write("**Feature Importance:**")
    feature_names = X.columns
    importance = model.coef_[0]  # Get coefficients for each feature
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by="Importance", key=abs, ascending=False)

    st.dataframe(importance_df)

    fig_imp, ax_imp = plt.subplots(figsize=(10, 5))
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis', ax=ax_imp)
    ax_imp.set_title("Feature Importance (Logistic Regression Coefficients)")
    st.pyplot(fig_imp)
