import pandas as pd
import numpy as np
import json
from pandas.api.types import is_numeric_dtype, is_integer_dtype
from openai import OpenAI

# 🔑 Multiple API keys: Mansi + Yours
OPENROUTER_KEYS = [
    "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",  # Mansi’s
    "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"   # Yours
]

# 🔁 Rotating GPT client
class GPTClientRotator:
    def __init__(self, api_keys, base_url="https://openrouter.ai/api/v1"):
        self.api_keys = api_keys
        self.base_url = base_url
        self.index = 0

    def get_client(self):
        return OpenAI(api_key=self.api_keys[self.index], base_url=self.base_url)

    def rotate_key(self):
        self.index += 1
        if self.index >= len(self.api_keys):
            raise Exception("❌ All API keys exhausted or failed.")

    def chat_completion(self, **kwargs):
        while True:
            try:
                client = self.get_client()
                return client.chat.completions.create(**kwargs)
            except Exception as e:
                self.rotate_key()

# 🧠 Initialize rotator
rotator = GPTClientRotator(OPENROUTER_KEYS)

# 🔍 Column verifier using GPT
def verify_with_gpt(col_name, sample_values, local_guess):
    prompt = (
        f"I am analyzing a dataset. One column is named '{col_name}', "
        f"and it contains these values: {sample_values}. "
        f"What type of data does this column contain?\n"
        f"Choose from: Time, Numeric, Ordinal, Category, ID, Free Text.\n"
        f"My program thinks it's: {local_guess}. Just reply with only the type."
    )
    try:
        response = rotator.chat_completion(
            model="mistralai/mistral-7b-instruct:free",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Final GPT error (all keys failed) for column '{col_name}':", e)
        return None


# 📥 Load dataset
df = pd.read_csv("input.csv")
print("Preview of Data:")
print(df.head())

# ✅ Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
print("✅ Column names standardized")

# 🚫 Handle missing values
print("\nMissing values (column-wise):")
print(df.isnull().sum())
for col in df.columns[df.isnull().any()]:
    if is_numeric_dtype(df[col]):
        mean_val = df[col].mean()
        mean_val = round(mean_val) if is_integer_dtype(df[col]) else round(mean_val, 2)
        df[col] = df[col].fillna(mean_val)
        print(f"Filled missing values in numeric column '{col}' with mean: {mean_val}")
    else:
        df[col] = df[col].fillna("Unknown")
        print(f"Filled missing values in text column '{col}' with 'Unknown'")

# 🧹 Remove duplicates
print(f"Total rows before removing duplicates: {df.shape[0]}")
duplicates = df.duplicated().sum()
df.drop_duplicates(inplace=True)
print(f"Removed {duplicates} duplicates. Remaining rows: {df.shape[0]}")

# 🔁 Convert types based on column names
print("Current column data types:\n", df.dtypes)
for col in df.columns:
    if df[col].dtype == "object":
        if "date" in col.lower() or "year" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce")
                print(f"Converted '{col}' to datetime")
            except:
                print(f"Could not convert '{col}' to datetime")
        else:
            df[col] = df[col].astype("category")
            print(f"Converted '{col}' to category")

# 🧠 GPT-Based Type Verification
print("\n🔍 Verifying column types with GPT...")
final_types = {}
for col in df.columns:
    non_null_values = df[col].dropna()
    if len(non_null_values) == 0:
        print(f"⚠️ Skipping GPT check for column '{col}' — no non-null values")
        final_types[col] = "datetime" if col.lower().endswith("date") or col.lower().endswith("year") else "Unknown"
        continue

    sample_values = non_null_values.sample(min(5, len(non_null_values))).tolist()
    unique = df[col].nunique()
    dtype = df[col].dtype

    # Your program's type guess
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        local_guess = "Time"
    elif "year" in col.lower() and df[col].dropna().apply(lambda x: str(x).isdigit() and 1000 <= int(x) <= 2100).all():
        local_guess = "Time"
    elif unique == len(df[col]):
        local_guess = "ID"
    elif dtype in ['int64', 'float64'] and unique <= 15:
        local_guess = "Ordinal"
    elif dtype in ['int64', 'float64']:
        local_guess = "Numeric"
    elif dtype.name == 'category' and unique <= 15:
        local_guess = "Category"
    else:
        local_guess = "Free Text"
     # 🎯 Filtering: Decide if this column really needs GPT verification
    should_verify = False  # Default to skipping unless conditions met

    # Skip if datetime
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        should_verify = False

    # Skip if column name has "date" or "year" and it's parsed as datetime
    elif ("date" in col.lower() or "year" in col.lower()) and pd.api.types.is_datetime64_any_dtype(df[col]):
        should_verify = False

    # Skip if unique = total rows (likely ID)
    elif unique == len(df[col]):
        should_verify = False

    # ✅ Check if logic might be wrong — these are edge cases
    elif dtype in ['int64', 'float64'] and 5 < unique <= 25:
        should_verify = True  # Could be ordinal or numeric — confusing

    elif dtype.name == 'category' and 10 < unique <= 30:
        should_verify = True  # Could be free text vs. category

    elif dtype.name == 'category' and unique > 30:
        should_verify = False  # Too many values — assume free text or ID

    elif dtype in ['int64', 'float64'] and unique <= 5:
        should_verify = True  # Could be ordinal — worth verifying

    elif dtype == object and unique <= 5:
        should_verify = True  # Could be Category/Ordinal — check

    else:
        should_verify = False  # Everything else — trust logic

    if not should_verify:
        final_types[col] = "datetime" if local_guess.lower() == "time" else local_guess
        print(f"🧠 Skipped GPT for column '{col}', used local guess: {local_guess}")
        continue

    gpt_guess = verify_with_gpt(col, sample_values, local_guess)

    if gpt_guess:
        gpt_guess = gpt_guess.split()[0].strip().capitalize()
        if gpt_guess.lower() != local_guess.lower():
            print(f"⚠️ Mismatch for '{col}': Code guessed '{local_guess}', GPT guessed '{gpt_guess}'")
            user_input = input(f"Do you agree with GPT’s guess '{gpt_guess}' for column '{col}'? (yes/no): ")
            if user_input.lower() == "no":
                correct = input("Enter correct type (Time, Numeric, Ordinal, Category, ID, Free Text): ")
                if correct.lower() == "time":
                    try:
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                        print(f"⏳ Converted '{col}' to datetime")
                        final_types[col] = "datetime"
                    except Exception as e:
                        print(f"❌ Failed to convert '{col}' to datetime:", e)
                        final_types[col] = "Time"
                else:
                    final_types[col] = correct
                print(f"✅ Column '{col}' final type set to: {final_types[col]}")
            else:
                if gpt_guess.lower() == "time":
                    try:
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                        print(f"⏳ Converted '{col}' to datetime")
                        final_types[col] = "datetime"
                    except Exception as e:
                        print(f"❌ Failed to convert '{col}' to datetime:", e)
                        final_types[col] = "Time"
                else:
                    final_types[col] = gpt_guess
                print(f"✅ Confirmed GPT guess for '{col}'")
        else:
            final_types[col] = "datetime" if local_guess.lower() == "time" else local_guess
            print(f"✅ Verified column '{col}' as '{local_guess}'")
    else:
        final_types[col] = "datetime" if local_guess.lower() == "time" else local_guess
        print(f"⚠️ GPT unavailable. Used local guess for column '{col}'")


# 💾 Save final results
with open("final_column_types.json", "w") as f:
    json.dump(final_types, f, indent=2)
    print("✅ Saved verified column types to final_column_types.json")



with open("final_column_types.json", "r") as f:
    column_types = json.load(f)
    category_cols =[]
    ordinal_cols = []
    for col,col_type in column_types.items():
        if col_type == "Category":
            category_cols.append(col)
        elif col_type == "Ordinal":
            ordinal_cols.append(col) 

print("Category columns: ", category_cols)
print("Ordinal columns: ", ordinal_cols)               

# 🚨 Final Missing Value Check after conversions
for col in df.columns[df.isnull().any()]:
    if is_numeric_dtype(df[col]):
        mean_val = df[col].mean()
        df[col] = df[col].fillna(mean_val)
        print(f"[FINAL] Filled missing values in numeric column '{col}' with mean: {mean_val}")
    else:
        df[col] = df[col].fillna("Unknown")
        print(f"[FINAL] Filled missing values in text column '{col}' with 'Unknown'")



df.to_csv("cleaned_input.csv", index=False)
print("✅ Cleaned data saved to 'cleaned_input.csv'")
