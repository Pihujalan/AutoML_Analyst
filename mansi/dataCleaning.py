import pandas as pd
import numpy as np
import json
from pandas.api.types import is_numeric_dtype, is_integer_dtype
from openai import OpenAI
from openai import OpenAI
import re
import re

def classify_id_column(col, series, total_rows):
    col_lower = col.lower()
    id_terms = ["id", "uuid", "ssn", "guid", "key", "account", "registration", "passport", "record"]
    col_compact = re.sub(r'[\s_]', '', col_lower)
    if any(term in col_compact for term in id_terms):
        return True
    nunique = series.nunique(dropna=True)
    if (nunique >= 0.985 * total_rows):
        money_keywords = ["amount", "price", "pay", "total", "revenue", "cost", "charge"]
        year_keywords = ["date", "dob", "year", "time", "timestamp"]
        counter_keywords = ["count", "number", "quantity"]
        if any(term in col_lower for term in money_keywords + year_keywords + counter_keywords):
            return False
        if series.dtype in ['int64', 'float64']:
            if series.diff().abs().mean() <= 2:
                return False
        if series.dtype == object and series.str.len().std() < 2 and series.str.len().mean() in (8, 16, 32, 36):
            return True
        return True
    return False

def classify_name_column(series, sample_size=25, threshold=0.8):
    sample = series.dropna().astype(str).sample(
        min(sample_size, max(1, series.notnull().sum()))
    )
    def check_name(value):
        words = value.split()
        if len(words) != 2:
            return False
        return all(w.isalpha() and w[0].isupper() for w in words)
    ratio = np.mean([check_name(val) for val in sample])
    return ratio >= threshold


client = OpenAI(
    api_key="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",
    base_url="https://api.groq.com/openai/v1"
)
''''
# 🔑 Multiple API keys: Mansi + Yours
OPENROUTER_KEYS = [
    "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",  # Mansi’s
    "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",   # Yours
    "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",  # User provided
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
'''
# 🧠 Initialize rotator
#rotator = GPTClientRotator(OPENROUTER_KEYS)

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
        #response = rotator.chat_completion(
            #model="mistralai/mistral-7b-instruct:free",
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Final GPT error (all keys failed) for column '{col_name}':", e)
        return None


# 📥 Load dataset
#df = pd.read_csv("input.csv")
#print("Preview of Data:")
#print(df.head())

# ✅ Standardize column names
#df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
#print("✅ Column names standardized")

# 🚫 Handle missing values
#print("\nMissing values (column-wise):")
#print(df.isnull().sum())
# 🧠 Force integer type based on column name
force_int_keywords = ["age", "year", "count", "number", "quantity"]

def clean_data_with_logs(df, log_path="logs/cleaning_log.txt", st_container=None, user_confirmed_types=None):
    import os
    mismatches = []
    gpt_failed = False
    user_types_dict = {}
    if user_confirmed_types is not None:
        user_types_dict = {item['column']: item['user_type'] for item in user_confirmed_types}
    final_types = {}
    logic_guesses = {}
    sample_values_dict = {}
    # --- Usual cleaning logic up to type inference ---
    for col in df.columns:
        col_lower = col.lower()
        if is_numeric_dtype(df[col]):
            if any(key in col_lower for key in force_int_keywords):
                if df[col].isnull().any():
                    mean_val = round(df[col].mean())
                    df[col] = df[col].fillna(mean_val)
                try:
                    df[col] = df[col].astype(int)
                except:
                    pass
    df.drop_duplicates(inplace=True)
    for col in df.columns:
        if df[col].dtype.name == 'category':
            if 'Unknown' not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories(['Unknown'])
            df[col] = df[col].fillna('Unknown')
        elif df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown')
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            df[col] = df[col].astype(str).str.strip().str.title()
    for col in df.columns:
        if df[col].dtype == "object":
            datetime_keywords = ["date", "year", "dob", "birth", "timestamp", "time", "created", "modified", "registered"]
            if any(key in col.lower() for key in datetime_keywords):
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    print(f"Converted '{col}' to datetime")
                except Exception as e:
                    print(f"Could not convert '{col}' to datetime: {e}")
                    df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce")
                except:
                    pass
            else:
                df[col] = df[col].astype("category")
    # --- Type inference and mismatch collection ---
    for col in df.columns:
        non_null_values = df[col].dropna()
        if len(non_null_values) == 0:
            final_types[col] = "datetime" if col.lower().endswith("date") or col.lower().endswith("year") else "Unknown"
            logic_guesses[col] = final_types[col]
            sample_values_dict[col] = []
            continue
        sample_values = non_null_values.sample(min(5, len(non_null_values))).tolist()
        unique = df[col].nunique()
        dtype = df[col].dtype
        # Local guess
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            logic_guess = "Time"
        elif "year" in col.lower() and df[col].dropna().apply(lambda x: str(x).isdigit() and 1000 <= int(x) <= 2100).all():
            logic_guess = "Time"
        elif unique == len(df[col]):
            logic_guess = "ID"
        elif dtype in ['int64', 'float64'] and unique <= 15:
            logic_guess = "Ordinal"
        elif dtype in ['int64', 'float64']:
            logic_guess = "Numeric"
        elif dtype.name == 'category' and unique <= 15:
            logic_guess = "Category"
        else:
            logic_guess = "Free Text"
        logic_guesses[col] = logic_guess
        sample_values_dict[col] = sample_values
        should_verify = False
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            should_verify = False
        elif ("date" in col.lower() or "year" in col.lower()) and pd.api.types.is_datetime64_any_dtype(df[col]):
            should_verify = False
        elif unique == len(df[col]):
            should_verify = False
        elif dtype in ['int64', 'float64'] and 5 < unique <= 25:
            should_verify = True
        elif dtype.name == 'category' and 10 < unique <= 30:
            should_verify = True
        elif dtype.name == 'category' and unique > 30:
            should_verify = False
        elif dtype in ['int64', 'float64'] and unique <= 5:
            should_verify = True
        elif dtype == object and unique <= 5:
            should_verify = True
        else:
            should_verify = False
        # --- User override: set type and cast column ---
        if col in user_types_dict:
            user_type = user_types_dict[col]
            final_types[col] = user_type
            # Cast column to user type
            if user_type.lower() == "numeric":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif user_type.lower() == "ordinal":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif user_type.lower() == "category":
                df[col] = df[col].astype("category")
            elif user_type.lower() == "time" or user_type.lower() == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif user_type.lower() == "id":
                df[col] = df[col].astype(str)
            elif user_type.lower() == "free text":
                df[col] = df[col].astype(str)
            continue
        # --- No user override, check for mismatch ---
        if not should_verify:
            final_types[col] = "datetime" if logic_guess.lower() == "time" else logic_guess
            continue
        try:
            gpt_guess = verify_with_gpt(col, sample_values, logic_guess)
        except Exception:
            gpt_failed = True
            gpt_guess = None
        if gpt_guess is None:
            gpt_failed = True
        if gpt_guess:
            gpt_guess = gpt_guess.split()[0].strip().capitalize()
            '''
            if gpt_guess.lower() != logic_guess.lower():
                print(f"⚠️ Mismatch for '{col}': Code guessed '{logic_guess}', GPT guessed '{gpt_guess}'")
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
                           final_types[col]= "datetime"
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
                           final_types[col] = "datetime"
                    else:
                        final_types[col] = gpt_guess
                    print(f"✅ Confirmed GPT guess for '{col}'")'''
            if gpt_guess.lower() != logic_guess.lower():
                mismatches.append({
                    "column": col,
                    "logic_guess": logic_guess,
                    "gpt_guess": gpt_guess,
                    "sample_values": sample_values
                })
                continue
            else:
                final_types[col] = "datetime" if logic_guess.lower() == "time" else logic_guess
        else:
            final_types[col] = "datetime" if logic_guess.lower() == "time" else logic_guess
    # --- If GPT failed for any column, require user to classify all columns ---
    if gpt_failed and not user_types_dict:
        columns_to_review = []
        for col in df.columns:
            columns_to_review.append({
                "column": col,
                "logic_guess": logic_guesses.get(col, "Unknown"),
                "gpt_guess": "Unavailable",
                "sample_values": sample_values_dict.get(col, [])
            })
        return df, columns_to_review
    # --- If mismatches exist and no user choices, return for UI ---
    if mismatches and not user_types_dict:
        return df, mismatches
    # --- Save user choices if provided ---
    if user_types_dict:
        with open("user_column_types.json", "w") as f:
            json.dump(user_types_dict, f, indent=2)
    # --- Fill in any columns not yet assigned (e.g., mismatches resolved now) ---
    for col in df.columns:
        if col not in final_types:
            if col in user_types_dict:
                final_types[col] = user_types_dict[col]
            else:
                non_null_values = df[col].dropna()
                unique = df[col].nunique()
                dtype = df[col].dtype
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    logic_guess = "Time"
                elif "year" in col.lower() and df[col].dropna().apply(lambda x: str(x).isdigit() and 1000 <= int(x) <= 2100).all():
                    logic_guess = "Time"
                elif unique == len(df[col]):
                    logic_guess = "ID"
                elif dtype in ['int64', 'float64'] and unique <= 15:
                    logic_guess = "Ordinal"
                elif dtype in ['int64', 'float64']:
                    logic_guess = "Numeric"
                elif dtype.name == 'category' and unique <= 15:
                    logic_guess = "Category"
                else:
                    logic_guess = "Free Text"
                final_types[col] = "datetime" if logic_guess.lower() == "time" else logic_guess
    # --- Save final types ---
    with open("final_column_types.json", "w") as f:
        json.dump(final_types, f, indent=2)
    # --- Continue with rest of cleaning logic (missing value fill, etc.) ---
    for col in df.columns[df.isnull().any()]:
        if is_integer_dtype(df[col]):
            mean_val = round(df[col].mean())
            df[col] = df[col].fillna(mean_val).astype(int)
        elif is_numeric_dtype(df[col]):
            non_null_values = df[col].dropna()
            if all(float(x).is_integer() for x in non_null_values):
                mean_val = round(df[col].mean())
                df[col] = df[col].fillna(mean_val).astype(int)
            else:
                mean_val = round(df[col].mean(), 2)
                df[col] = df[col].fillna(mean_val)
        else:
            df[col] = df[col].fillna("Unknown")
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/cleaned_input.csv", index=False)
    return df, []


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
        if is_integer_dtype(df[col]):
            if df[col].notnull().sum() > 0:
                mean_val = round(df[col].mean())
            else:
                mean_val = 0  # or a context-appropriate default
            df[col] = df[col].fillna(mean_val).astype(int)
            print(f"✅ Filled missing values in integer column '{col}' with rounded mean: {mean_val}")
        elif is_numeric_dtype(df[col]):
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0 and all(float(x).is_integer() for x in non_null_values):
                mean_val = round(df[col].mean())
                df[col] = df[col].fillna(mean_val).astype(int)
                print(f"✅ Filled numeric column '{col}' with rounded mean (int): {mean_val}")
            elif len(non_null_values) > 0:
                mean_val = round(df[col].mean(), 2)
                df[col] = df[col].fillna(mean_val)
                print(f"✅ Filled numeric column '{col}' with rounded mean (float): {mean_val}")
            else:
                df[col] = df[col].fillna(0.0)
                print(f"⚠️ Filled completely null numeric column '{col}' with zeros (float).")
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            if df[col].notnull().sum() > 0:
                fill_date = df[col].mode().iloc[0]
            else:
                fill_date = pd.to_datetime("2000-01-01")
            df[col] = df[col].fillna(fill_date)
            print(f"✅ Filled missing dates in '{col}' with '{fill_date}'")
        else:
            df[col] = df[col].fillna("Unknown")
            print(f"✅ Filled missing values in text column '{col}' with 'Unknown'")




    
    
        # --- DROP ID & NAME columns classified from final_types ---
    id_name_cols = [col for col, typ in final_types.items() if typ.lower() in ("id", "name")]
    if id_name_cols:
        print(f"🗑️ Dropping ID/Name columns: {id_name_cols}")
        df = df.drop(columns=id_name_cols)
    else:
        print("No ID or Name columns found to drop!")

    # --- Force-drop columns with 'id' or 'name' in their name (except product name etc) ---
    exclude_safe = ("product name", "product_id", "sub-category", "category", "segment")  # extend as needed

    cols_to_force_drop = []
    for col in df.columns:
        col_lower = col.lower().replace("_", " ").strip()
        # Drop if 'id' or 'name' in column, unless it's in exclude_safe
        if (("id" in col_lower or "name" in col_lower) and
            not any(safe in col_lower for safe in exclude_safe)):
            cols_to_force_drop.append(col)

    if cols_to_force_drop:
        print(f"Force-dropping these ID/name columns: {cols_to_force_drop}")
        df = df.drop(columns=cols_to_force_drop)
    df.to_csv("cleaned_input.csv", index=False)
    print("✅ Cleaned data saved to 'cleaned_input.csv'")
    
    return df
