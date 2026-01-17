import sqlite3
import pandas as pd

# Connect or create DB
conn = sqlite3.connect("mediscan.db")

# Read all CSVs
df_diseases = pd.read_csv("D:\MediScannn\ML-API\dataset.csv")
df_precautions = pd.read_csv("D:\MediScannn\ML-API\symptom_precaution.csv")
df_descriptions = pd.read_csv("D:\MediScannn\ML-API\symptom_Description.csv")
df_severity = pd.read_csv("D:\MediScannn\ML-API\Symptom-severity.csv")

# Save each CSV as a table
df_diseases.to_sql("diseases", conn, if_exists="replace", index=False)
df_precautions.to_sql("precautions", conn, if_exists="replace", index=False)
df_descriptions.to_sql("descriptions", conn, if_exists="replace", index=False)
df_severity.to_sql("symptoms", conn, if_exists="replace", index=False)

print("✅ Database created as mediscan.db")
conn.commit()
conn.close()
