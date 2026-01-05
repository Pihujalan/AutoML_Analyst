import pandas as pd
import os
from sklearn.model_selection import train_test_split

# ---------- Load CSV ----------
# Try new location first, then fallback to old
if os.path.exists("outputs/data/encoded_input.csv"):
    df = pd.read_csv("outputs/data/encoded_input.csv")
elif os.path.exists("data_processing/encoded_input.csv"):
    df = pd.read_csv("data_processing/encoded_input.csv")
elif os.path.exists("encoded_input.csv"):
    df = pd.read_csv("encoded_input.csv")
else:
    raise FileNotFoundError("encoded_input.csv not found in any expected location")
print("Available columns in the dataset: ")
print(df.columns.tolist())

# ---------- Ask for target column ----------
target_col = input("\nEnter the name of the target column from the above list: ")
if target_col not in df.columns:
    print(f"❌ Column '{target_col}' not found in the dataset. Exiting.")
    exit()

X = df.drop(columns=[target_col])
Y = df[target_col]
print(f"\n✅ Features and target separated. Shape of X: {X.shape}, Y: {Y.shape}")

# ---------- Smart Stratified Split Function ----------
def smart_train_test_split(X, Y, test_size=0.2, random_state=42):
    stratify_col = None
    if Y.dtype == 'object' or Y.nunique() <= 10:
        stratify_col = Y
        print(f"✅ Stratifying on categorical target column: '{Y.name}'")
    else:
        try:
            binned = pd.qcut(Y, q=5, duplicates='drop')
            stratify_col = binned
            print("✅ Stratifying on binned numeric target (quantiles)")
        except Exception as e:
            print(f"⚠️ Stratification failed: {e} — using random split instead")
            stratify_col = None

    return train_test_split(X, Y, test_size=test_size, random_state=random_state, stratify=stratify_col)

# ---------- Perform Split ----------
X_train, X_test, Y_train, Y_test = smart_train_test_split(X, Y)

# ---------- Save CSVs (preserve original format) ----------
os.makedirs("outputs/data", exist_ok=True)
X_train.to_csv("outputs/data/X_train.csv", index=False)
X_test.to_csv("outputs/data/X_test.csv", index=False)
Y_train.to_csv("outputs/data/Y_train.csv", index=False)
Y_test.to_csv("outputs/data/Y_test.csv", index=False)

print("\n✅ Data split completed and saved:")
print(f"   - X_train: {X_train.shape}")
print(f"   - X_test: {X_test.shape}")
print(f"   - Y_train: {Y_train.shape}")
print(f"   - Y_test: {Y_test.shape}")