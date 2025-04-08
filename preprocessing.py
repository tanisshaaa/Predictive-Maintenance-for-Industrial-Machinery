import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
df = pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\TANISHA\PROJECTS\Predictive Maintenance (PdM) for Industrial Machinery\ai4i2020.csv")  # Replace with your file path

# Rename columns to be Python/SQL-friendly
df.columns = [
    "UDI", "Product_ID", "Type", "Air_temperature_K", "Process_temperature_K",
    "Rotational_speed_rpm", "Torque_Nm", "Tool_wear_min",
    "Machine_failure", "TWF", "HDF", "PWF", "OSF", "RNF"
]

# Encode the 'Type' column
le = LabelEncoder()
df['Type'] = le.fit_transform(df['Type'])

# Handle missing values (if any)
if df.isnull().sum().any():
    df.fillna(df.median(numeric_only=True), inplace=True)

# Standardize numerical features
scaler = StandardScaler()
numerical_cols = [
    "Air_temperature_K", "Process_temperature_K",
    "Rotational_speed_rpm", "Torque_Nm", "Tool_wear_min"
]
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Save preprocessed data
df.to_csv("pdm_preprocessed.csv", index=False)

print("✅ Data preprocessing complete. Saved as 'pdm_preprocessed.csv'.")
