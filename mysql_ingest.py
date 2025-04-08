import pandas as pd
import mysql.connector

# Load preprocessed data
df = pd.read_csv("pdm_preprocessed.csv")

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="tanitani",
    database="pdm_db"
)

cursor = conn.cursor()

# Optional: create the table (only once)
create_table_query = '''
CREATE TABLE IF NOT EXISTS predictive_maintenance (
    UDI INT,
    Product_ID VARCHAR(20),
    Type INT,
    Air_temperature_K FLOAT,
    Process_temperature_K FLOAT,
    Rotational_speed_rpm FLOAT,
    Torque_Nm FLOAT,
    Tool_wear_min FLOAT,
    Machine_failure INT,
    TWF INT,
    HDF INT,
    PWF INT,
    OSF INT,
    RNF INT
)
'''
cursor.execute(create_table_query)

# Insert rows from DataFrame
for _, row in df.iterrows():
    insert_query = '''
    INSERT INTO predictive_maintenance (
        UDI, Product_ID, Type, Air_temperature_K, Process_temperature_K,
        Rotational_speed_rpm, Torque_Nm, Tool_wear_min,
        Machine_failure, TWF, HDF, PWF, OSF, RNF
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    '''
    cursor.execute(insert_query, tuple(row))

conn.commit()
cursor.close()
conn.close()

print("✅ Data inserted into MySQL database successfully.")
