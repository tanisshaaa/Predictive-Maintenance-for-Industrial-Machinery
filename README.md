# 🛠️ Predictive Maintenance for Industrial Machinery

## 📌 Why is it Necessary?

Predictive Maintenance (PdM) plays a critical role in industrial operations. Rather than relying on reactive or scheduled maintenance, PdM uses real-time data and machine learning to **predict failures before they occur**, thereby:
- Reducing unexpected downtimes
- Lowering maintenance costs
- Improving safety and efficiency
- Extending equipment lifespan

---

## 📊 Columns Used (Features)

The dataset used in this project consists of the following features:

- `MachineID`: Unique identifier for each machine
- `Voltage`, `Rotation`, `Pressure`, `Vibration`: Sensor readings
- `TimeSinceMaintenance`: Time since the last maintenance check
- `Temperature`: Ambient or machine temperature
- `Failure`: Binary target variable indicating failure (1) or normal operation (0)

---

## 🚀 Streamlit Website
You can explore the interactive dashboard and make predictions using the live Streamlit app:

👉 Live App on Streamlit [(https://predictive-maintenance-for-industrial-machinery.streamlit.app/)]

### 1. 📊 Features of the Website:
Exploratory Data Analysis (EDA):
Interactive visualizations and correlation plots to understand data trends and relationships.

### 2. Machine Learning Model Performance:
Visual reports showing model evaluation metrics like accuracy, precision, recall, ROC curves, and more.

### 3. 🔮 Real-Time Failure Prediction:
Input custom feature values and get instant predictions on potential machine failure within 24 hours.


## 🤖 Models Used & Why

Three models were implemented for a comprehensive analysis:

### 1. **Linear Regression**
- **Purpose:** Understand the regression behavior between variables and predicted values.
- **Why:** Acts as a baseline model and helps visualize relationships for continuous target values.

### 2. **Random Forest Classifier**
- **Purpose:** Classification of failure (0/1).
- **Why:** Excellent for handling non-linear data and avoids overfitting due to ensemble nature.

### 3. **Logistic Regression**
- **Purpose:** Binary classification for predicting failure probability.
- **Why:** Simple and interpretable model often used for binary outcomes with well-calibrated probabilities.

---

## 📦 Dependencies and Libraries Used

Below are the key dependencies used in this project:

- `pandas`: For data manipulation and preprocessing  
- `numpy`: For numerical operations  
- `matplotlib` & `seaborn`: For visualization  
- `scikit-learn`: For modeling, metrics, and splitting the data  
- `streamlit`: To create and host the interactive web app  
- `joblib`: For model serialization  
- `mysql-connector-python` *(optional)*: If using a database instead of CSV

You can install all dependencies using:

```bash
pip install -r requirements.txt
```

---

Let me know if you want to add **screenshots**, **badges**, or a **demo video link** too!

