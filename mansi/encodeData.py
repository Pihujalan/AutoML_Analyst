import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("cleaned_input.csv")

print("Original shape:", df.shape)

# Ask for target column (not needed for label encoding, but kept for compatibility)
print("\nAvailable columns:")
print(df.columns.tolist())
target_col = input("\nEnter the name of the target column (used only for Target Encoding): ")

if target_col not in df.columns:
    print(f"❌ Target column '{target_col}' not found in the dataset. Exiting.")
    exit()

# Save the target column
with open("target_config.json", "w") as f:
    json.dump({"target_column": target_col}, f)

# --- STEP 1: Convert all boolean-like columns to 0/1 ---
for col in df.columns:
    unique_vals = set(df[col].dropna().unique())
    if unique_vals <= {True, False} or unique_vals <= {'True', 'False'}:
        df[col] = df[col].replace({True: 1, False: 0, 'True': 1, 'False': 0}).astype(int)

# --- STEP 2: Label encode all object/category columns ---
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# --- STEP 3: Save final encoded output ---
df.to_csv("encoded_input.csv", index=False)
print("\n✅ Label encoding complete. Output saved as 'encoded_input.csv'")
