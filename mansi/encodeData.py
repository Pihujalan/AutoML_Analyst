import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import json

# Load dataset
df = pd.read_csv("input.csv")
print("Original shape:", df.shape)

# Ask for target column
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

# --- STEP 2: Identify categorical columns (object/category) ---
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

# --- STEP 3: Generate encoding suggestions ---
encoding_suggestions = {}
for col in cat_cols:
    unique_count = df[col].nunique()
    if unique_count <= 10:
        suggestion = "onehot"
    elif unique_count <= 50:
        suggestion = "frequency"
    else:
        suggestion = "target"
    encoding_suggestions[col] = {
        "unique_values": unique_count,
        "recommended_encoding": suggestion
    }

print("\nEncoding Suggestions:")
print(json.dumps(encoding_suggestions, indent=4))

# --- STEP 4: Define encoding functions ---
def apply_onehot_encoding(df, col):
    return pd.get_dummies(df, columns=[col], drop_first=True)

def apply_frequency_encoding(df, col):
    freq_map = df[col].value_counts().to_dict()
    df[col] = df[col].map(freq_map)
    return df

def apply_target_encoding(df, col, target):
    means = df.groupby(col)[target].mean().to_dict()
    df[col] = df[col].map(means)
    return df

# --- STEP 5: Apply encoding ---
encoded_df = df.copy()
for col, info in encoding_suggestions.items():
    method = info["recommended_encoding"]
    print(f"\n🔄 Encoding '{col}' using {method}...")

    if method == "onehot":
        encoded_df = apply_onehot_encoding(encoded_df, col)
    elif method == "frequency":
        encoded_df = apply_frequency_encoding(encoded_df, col)
    elif method == "target":
        if target_col in encoded_df.columns:
            encoded_df = apply_target_encoding(encoded_df, col, target_col)
        else:
            print(f"❌ Skipping Target Encoding for '{col}' — target column not found.")

# --- STEP 6: FINAL CHECK - Convert any remaining boolean-like columns to 0/1 ---
for col in encoded_df.columns:
    unique_vals = set(encoded_df[col].dropna().unique())
    if unique_vals <= {True, False} or unique_vals <= {'True', 'False'}:
        encoded_df[col] = encoded_df[col].replace({True: 1, False: 0, 'True': 1, 'False': 0}).astype(int)

# --- STEP 7: Save final encoded output ---
encoded_df.to_csv("encoded_input.csv", index=False)
print("\n✅ Encoding complete. Output saved as 'encoded_input.csv'")


