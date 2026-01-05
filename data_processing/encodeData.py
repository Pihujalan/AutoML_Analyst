import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Ensure outputs directory structure exists
os.makedirs("outputs/encoders", exist_ok=True)
os.makedirs("outputs/data", exist_ok=True)

# Load dataset
#df = pd.read_csv("cleaned_input.csv")
df = pd.read_csv("insurance.csv")
print("Original shape:", df.shape)

# Target column input (only used for Target Encoding, not applied here)
print("\nAvailable columns:")
print(df.columns.tolist())
target_col = input("\nEnter the name of the target column (used only for Target Encoding): ")

if target_col not in df.columns:
    print(f"❌ Target column '{target_col}' not found in the dataset. Exiting.")
    exit()

# Save the target column config
os.makedirs("outputs/data", exist_ok=True)
with open("outputs/data/target_config.json", "w") as f:
    json.dump({"target_column": target_col}, f)

# --- STEP 1: Convert boolean-like columns to 0/1 ---
for col in df.columns:
    unique_vals = set(df[col].dropna().unique())
    if unique_vals <= {True, False} or unique_vals <= {'True', 'False'}:
        df[col] = df[col].replace({True: 1, False: 0, 'True': 1, 'False': 0}).astype(int)

# --- STEP 2: Manually normalize key binary columns ---
if 'gender' in df.columns:
    df['gender'] = df['gender'].astype(str).str.lower().str.strip()
    df['gender'] = df['gender'].replace({
        'male': 1, 'm': 1,
        'female': 0, 'f': 0
    })

if 'everbenched' in df.columns:
    df['everbenched'] = df['everbenched'].astype(str).str.lower().str.strip()
    df['everbenched'] = df['everbenched'].replace({
        'yes': 1, 'y': 1,
        'no': 0, 'n': 0
    })

# --- STEP 3: Label encode remaining object/category columns ---
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_cols:
    df[col] = df[col].astype(str).str.lower().str.strip()  # Clean text
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    # Save encoders to outputs/encoders directory
    encoders_dir = "outputs/encoders"
    os.makedirs(encoders_dir, exist_ok=True)

    # Save encoder to correct folder
    joblib.dump(le, os.path.join(encoders_dir, f"{col}_encoder.pkl"))
# --- STEP 4: Save final encoded data ---
os.makedirs("outputs/data", exist_ok=True)
df.to_csv("outputs/data/encoded_input.csv", index=False)
print("\n✅ Encoding complete. Output saved as 'encoded_input.csv'")
print("🧠 LabelEncoders saved in 'encoders/' folder.")