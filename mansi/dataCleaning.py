import pandas as pd
import numpy as np
import json
from pandas.api.types import is_numeric_dtype, is_integer_dtype
from openai import OpenAI

import re


def detect_date_format(sample_dates):
    european, us = 0, 0
    for date_str in sample_dates:
        try:
            # Standardize all separators to "-"
            standardized = re.sub(r"[/.]", "-", date_str)
            parts = standardized.split("-")
            if len(parts) == 3:
                d1, d2, d3 = parts
                d1, d2, d3 = int(d1), int(d2), int(d3)
                if d1 > 12:
                    european += 1
                elif d2 > 12:
                    us += 1
                else:
                    # Ambiguous, default to US
                    us += 1
        except:
            continue
    return "euro" if european > us else "us"

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
    
    datetime_keywords = ["date", "year", "dob", "birth", "timestamp", "time", "created", "modified", "registered"]

    for col in df.columns:
        # Clean and convert date-like object columns
        if df[col].dtype == "object" and any(key in col.lower() for key in datetime_keywords):
            # Standardize separators and strip whitespace
            df[col] = df[col].astype(str).str.strip().str.replace(r"[/.]", "-", regex=True)
            df[col] = df[col].replace(
                ["None", "none", "", "nan", "NaN", "unknown", "Unknown"], np.nan
            )
            df[col] = df[col].where(df[col].notna(), np.nan)  # Ensure Python None is also np.nan
            sample_values = df[col].dropna().astype(str).head(10).tolist()
            # Try parsing with both dayfirst True and False
            parsed_dayfirst = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            parsed_monthfirst = pd.to_datetime(df[col], errors="coerce", dayfirst=False)
            # Choose the parse with fewer NaT (missing) values
            if parsed_dayfirst.notna().sum() >= parsed_monthfirst.notna().sum():
                df[col] = parsed_dayfirst
                print(f"✅ Converted '{col}' to datetime with dayfirst=True (auto-selected)")
            else:
                df[col] = parsed_monthfirst
                print(f"✅ Converted '{col}' to datetime with dayfirst=False (auto-selected)")
            print(f"🛑 Unparsed values in '{col}':", df[col][df[col].isna()].unique().tolist()[:10])
        # Handle stringy/categorical columns (do NOT touch date columns here!)
        elif df[col].dtype == "object" and not any(key in col.lower() for key in datetime_keywords):
            df[col] = df[col].fillna('Unknown')
            df[col] = df[col].astype(str).str.strip().str.title()
        elif df[col].dtype.name == 'category':
            if 'Unknown' not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories(['Unknown'])
            df[col] = df[col].fillna('Unknown')
            df[col] = df[col].astype(str).str.strip().str.title()

            
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
                elif unique >= 0.99 * len(df[col]):  # Very high threshold: 99% uniqueness
                    # Smart detection based on data characteristics, not hardcoded keywords
                    if dtype in ['int64', 'float64']:
                        # FIRST: Check if it has decimal places (decimal values are almost never IDs)
                        sample_values = df[col].dropna().sample(min(10, len(df[col])))
                        has_decimals = any(float(val) % 1 != 0 for val in sample_values if pd.notna(val))
                        if has_decimals:
                            logic_guess = "Numeric"  # Decimal values are almost never IDs
                        else:
                            # Check if it looks like a sequential ID (small differences between consecutive values)
                            if df[col].diff().abs().mean() <= 2:
                                logic_guess = "ID"
                            else:
                                # Check if values are in a reasonable range for IDs vs numeric features
                                min_val = df[col].min()
                                max_val = df[col].max()
                                if min_val >= 0 and max_val <= 999999:  # Reasonable ID range
                                    logic_guess = "ID"
                                else:
                                    logic_guess = "Numeric"  # Large numbers are usually features
                    else:
                        # For non-numeric columns with high uniqueness, check if they look like IDs
                        sample_values = df[col].dropna().astype(str).sample(min(10, len(df[col])))
                        # Check if they have consistent length (like IDs)
                        lengths = [len(str(val)) for val in sample_values]
                        if len(set(lengths)) <= 2:  # Consistent length
                            logic_guess = "ID"
                        else:
                            logic_guess = "Free Text"  # Variable length = likely not ID
                elif dtype in ['int64', 'float64'] and unique <= 15:
                    logic_guess = "Ordinal"
                elif dtype in ['int64', 'float64']:
                    logic_guess = "Numeric"
                elif dtype.name == 'category' and unique <= 15:
                    logic_guess = "Category"
                else:
                    logic_guess = "Free Text"
                final_types[col] = "datetime" if logic_guess.lower() == "time" else logic_guess
    
    # --- Enhanced ID and Name detection (BEFORE saving final types) ---
    def detect_id_or_name_column(col, series, col_type):
        """Enhanced detection for ID and Name columns"""
        col_lower = col.lower()
        
        # Override ID classification for decimal columns (decimal values are almost never IDs)
        if col_type.lower() == "id" and series.dtype in ['int64', 'float64']:
            sample_values = series.dropna().sample(min(10, len(series)))
            has_decimals = any(float(val) % 1 != 0 for val in sample_values if pd.notna(val))
            if has_decimals:
                return "Numeric"  # Decimal values are almost never IDs
            return "ID"
        
        # Check for name patterns (only if not already classified as something else)
        if col_type.lower() not in ["id", "name"]:
            # Check if values actually look like names (contain spaces, are title case, etc.)
            if series.dtype == 'object':
                sample_values = series.dropna().astype(str).sample(min(10, len(series)))
                name_like_count = 0
                for val in sample_values:
                    if ' ' in val and val.replace(' ', '').isalpha() and val.istitle():
                        name_like_count += 1
                if name_like_count / len(sample_values) > 0.7:
                    return "Name"
        
        # Additional check for high uniqueness (98% threshold) - no hardcoded keywords
        total_rows = len(series)
        nunique = series.nunique(dropna=True)
        if nunique >= 0.98 * total_rows:
            if series.dtype in ['int64', 'float64']:
                # Check if it's a sequential ID
                if series.diff().abs().mean() <= 2:
                    return "ID"
            elif series.dtype == object:
                # Check for UUID-like patterns
                if series.str.len().std() < 2 and series.str.len().mean() in (8, 16, 32, 36):
                    return "ID"
        
        return col_type
    
    # --- Apply enhanced detection to improve classifications ---
    ambiguous_columns = []
    for col in df.columns:
        if col in final_types:
            improved_type = detect_id_or_name_column(col, df[col], final_types[col])
            if improved_type != final_types[col]:
                print(f"🔄 Improved classification for '{col}': {final_types[col]} → {improved_type}")
                final_types[col] = improved_type
            
            # Collect potentially ambiguous ID columns for user confirmation
            if final_types[col] == "ID":
                # Only ask for confirmation if it's REALLY ambiguous:
                # 1. High uniqueness (99%+)
                # 2. No clear sequential pattern
                # 3. Not clearly an ID based on data characteristics
                is_high_uniqueness = df[col].nunique() >= 0.99 * len(df[col])
                
                if is_high_uniqueness:
                    # Check if it has clear sequential pattern (likely ID)
                    if df[col].dtype in ['int64', 'float64']:
                        is_sequential = df[col].diff().abs().mean() <= 2
                        if not is_sequential:
                            # Not sequential + high uniqueness = might be ambiguous
                            ambiguous_columns.append(col)
                    else:
                        # Non-numeric with high uniqueness = might be ambiguous
                        ambiguous_columns.append(col)
    
    # --- User confirmation for ambiguous ID columns ---
    if ambiguous_columns and st_container is None:  # Only prompt if not in Streamlit
        print(f"\n🤔 Found {len(ambiguous_columns)} columns classified as ID that might need review:")
        for col in ambiguous_columns:
            sample_values = df[col].dropna().sample(min(5, len(df[col]))).tolist()
            print(f"  - '{col}': {sample_values}")
            user_input = input(f"  Is '{col}' actually an ID column that should be dropped? (y/n): ")
            if user_input.lower() != 'y':
                print(f"  ✅ Keeping '{col}' as {final_types.get(col, 'Numeric')}")
                final_types[col] = "Numeric"  # Override to keep it
            else:
                print(f"  🗑️ Confirmed: '{col}' will be dropped as ID")
    
    # --- Streamlit interface for column confirmation ---
    elif ambiguous_columns and st_container is not None:
        st_container.write(f"🤔 Found {len(ambiguous_columns)} columns classified as ID that might need review:")
        for col in ambiguous_columns:
            sample_values = df[col].dropna().sample(min(5, len(df[col]))).tolist()
            st_container.write(f"  - '{col}': {sample_values}")
            if st_container.checkbox(f"Drop '{col}' as ID column?", value=True):
                st_container.write(f"  🗑️ '{col}' will be dropped")
            else:
                st_container.write(f"  ✅ '{col}' will be kept as Numeric")
                final_types[col] = "Numeric"
    
    # --- Save final types (AFTER enhanced detection) ---
    with open("final_column_types.json", "w") as f:
        json.dump(final_types, f, indent=2)
    
    # --- Identify columns to drop ---
    columns_to_drop = []
    for col in df.columns:
        if col in final_types and final_types[col] in ["ID", "Name"]:
            columns_to_drop.append(col)
            print(f"🗑️ Detected {final_types[col]} column: '{col}' - will be dropped")
    
    # --- Force-drop columns with 'id' or 'name' in their name (except safe ones) ---
    exclude_safe = ("product name", "product_id", "sub-category", "category", "segment", "region")
    
    for col in df.columns:
        if col not in columns_to_drop:  # Only check if not already marked for dropping
            col_lower = col.lower().replace("_", " ").strip()
            # Drop if 'id' or 'name' in column, unless it's in exclude_safe
            if (("id" in col_lower or "name" in col_lower) and
                not any(safe in col_lower for safe in exclude_safe)):
                columns_to_drop.append(col)
                print(f"🗑️ Force-dropping ID/name column: '{col}'")
    
    # --- Drop the identified columns ---
    if columns_to_drop:
        print(f"🗑️ Dropping {len(columns_to_drop)} ID/Name columns: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)
        # Update final_types to remove dropped columns
        for col in columns_to_drop:
            if col in final_types:
                del final_types[col]
        # Save updated types
        with open("final_column_types.json", "w") as f:
            json.dump(final_types, f, indent=2)
    else:
        print("✅ No ID or Name columns found to drop!")
    
    # --- Handle missing values after column dropping ---
    for col in df.columns[df.isnull().any()]:
        if is_integer_dtype(df[col]):
            if df[col].notnull().sum() > 0:
                mean_val = round(df[col].mean())
            else:
                mean_val = 0
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
    
    # --- Save cleaned data ---
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/cleaned_input.csv", index=False)
    print("✅ Cleaned data saved to 'data/cleaned_input.csv'")
    
    return df, []
