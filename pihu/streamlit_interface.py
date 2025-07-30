import streamlit as st
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.fullmodel import run_pattern_analysis_only, run_model_training_only
from mansi.dataCleaning import clean_data_with_logs
import json
import threading
import base64
import glob
import matplotlib.pyplot as plt
import seaborn as sns


def display_report(report_path):
    with open(report_path, "r") as file:
        content = file.read()

    st.subheader("📄 Insight Report")
    with st.container():
        st.markdown(content, unsafe_allow_html=True)  
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="final_pattern_report.txt">📥 Download Report as TXT</a>'
    st.markdown(href, unsafe_allow_html=True)

# Helper function to apply user types to DataFrame
def apply_user_types(df, user_types):
    for col, typ in user_types.items():
        if typ.lower() == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif typ.lower() == "category":
            df[col] = df[col].astype("category")
        elif typ.lower() == "date" or typ.lower() == "datetime":
            df[col] = pd.to_datetime(df[col], errors="coerce")
        elif typ.lower() == "id":
            # ID columns should be treated as strings to preserve their exact values
            df[col] = df[col].astype(str)
        elif typ.lower() in ["free text", "text"]:
            df[col] = df[col].astype(str)
    return df

# -------------------- Page Setup --------------------
st.set_page_config(page_title="AutoML Analyst", layout="wide")
st.title("AutoML Analyst")

# Initialize session state variables
if 'confirmed' not in st.session_state:
    st.session_state.confirmed = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'user_column_types' not in st.session_state:
    st.session_state.user_column_types = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'run_pipeline' not in st.session_state:
    st.session_state.run_pipeline = False
if 'show_analytics' not in st.session_state:
    st.session_state.show_analytics = False
if 'show_type_review' not in st.session_state:
    st.session_state.show_type_review = False
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False

# -------------------- Upload UI (if not confirmed) --------------------
if not st.session_state.confirmed:
    # Choose input method
    upload_method = st.radio(
        "Choose your data input method:",
        ["Upload CSV file", "Paste Google Sheet link"],
        index=0
    )

    # ----------- CSV Upload Option -----------
    if upload_method == "Upload CSV file":
        st.subheader("Upload your CSV file:")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df

            os.makedirs("data", exist_ok=True)
            df.to_csv("data/input.csv", index=False)

            st.success(" File uploaded and saved successfully.")
            st.subheader("Preview of your data:")
            st.dataframe(df.head(10))

            if st.button(" Yes, this is my dataset"):
                st.session_state.confirmed = True
                st.rerun()
            if st.button(" No, I want to upload again"):
                st.session_state.df = None
                st.rerun()
        else:
            st.info("Please upload a CSV file to proceed.")

    # ----------- Google Sheet Link Option -----------
    else:
        st.subheader("Paste your Google Sheet link:")

        with st.expander("How to load your Google Sheet into the app"):
            st.markdown("""
            1.Make sure your sheet has a proper **header row**
                - First row should have column names like: Name, Age, Gender, etc.

            2. Keep your data in the correct **tab (worksheet)**
                - For example: Sheet1, Data, or EmployeeInfo
                - You will need to enter the exact tab name in the app

            3. Share the Google Sheet with our bot email:
                📧 your-bot-service-account@your-project-id.iam.gserviceaccount.com
                - Open your sheet → Click "Share" → Paste the email above → Click "Send"

            4. Copy the **full Google Sheet link** from the browser address bar:
                Example: https://docs.google.com/spreadsheets/d/your-sheet-id/edit#gid=0

            5. In the app:
                - Paste the link in the input field
                - Type the sheet/tab name
                - Click "Load Sheet" to begin

            """)

        # Single-line clean input without browser suggestion
        sheet_url = st.text_input("Enter Google Sheet URL", value="", key="no_suggest_url")

        if sheet_url:
            try:
                if "docs.google.com" in sheet_url and "edit" in sheet_url:
                    # Convert to CSV link
                    csv_url = sheet_url.split("/edit")[0] + "/export?format=csv"
                    df = pd.read_csv(csv_url)
                    st.session_state.df = df

                    os.makedirs("data", exist_ok=True)
                    df.to_csv("data/input.csv", index=False)

                    st.success(" Google Sheet fetched and saved successfully.")
                    st.subheader("Preview of your data:")
                    st.dataframe(df.head(10))

                    if st.button(" Yes, this is my dataset"):
                        st.session_state.confirmed = True
                        st.rerun()
                    if st.button(" No, I want to upload again"):
                        st.session_state.df = None
                        st.rerun()
                else:
                    st.error("Invalid Google Sheet URL. Make sure it ends with /edit.")
            except Exception as e:
                st.error(f" Failed to load Google Sheet: {e}")

# -------------------- Confirmed Dataset --------------------
if st.session_state.confirmed and st.session_state.df is not None:
    st.success(" Dataset confirmed!")
    st.subheader("Your uploaded dataset:")
    st.dataframe(st.session_state.df.head(10))

    if st.button("back"):
        st.session_state.confirmed = False
        st.session_state.df = None
        st.rerun()

    # --- Handle type mismatches interactively ---
    if "user_confirmed_types" not in st.session_state:
        st.session_state.user_confirmed_types = None
    if "cleaning_result" not in st.session_state:
        # First run: no user choices yet
        df, mismatches, *_ = clean_data_with_logs(st.session_state.df, log_path="logs/cleaning_log.txt")
        st.session_state.cleaning_result = (df, mismatches)
    # If user just submitted choices, rerun cleaning with those
    if st.session_state.user_confirmed_types is not None:
        df, mismatches, *_ = clean_data_with_logs(st.session_state.df, log_path="logs/cleaning_log.txt", user_confirmed_types=st.session_state.user_confirmed_types)
        st.session_state.cleaned_df = df
        st.session_state.cleaning_result = (df, mismatches)
        st.session_state.user_confirmed_types = None
    df, mismatches = st.session_state.cleaning_result
    
    # Show cleaned data preview and download button immediately after cleaning
    st.subheader("Cleaned Data Preview")
    st.dataframe(df.head(10))
    
    # Download cleaned CSV button - show immediately after cleaning
    cleaned_path = "data/cleaned_input.csv"
    df.to_csv(cleaned_path, index=False)
    with open(cleaned_path, "rb") as f:
        st.download_button(
            "Download Cleaned CSV",
            f,
            file_name="cleaned_input.csv",
            mime="text/csv",
            help="Download the cleaned dataset",
            key="download_cleaned_csv"
        )
    
    st.markdown("---")
    
    # If mismatches, show form for user to resolve
    if mismatches:
        st.warning("Some columns have conflicting type guesses. Please choose the correct type for each:")
        with st.form("type_conflict_form"):
            user_choices = []
            for mismatch in mismatches:
                col = mismatch["column"]
                logic_guess = mismatch["logic_guess"]
                gpt_guess = mismatch["gpt_guess"]
                sample_values = mismatch["sample_values"]
                st.write(f"**Column:** `{col}`")
                st.write(f"Logic guess: `{logic_guess}` | GPT guess: `{gpt_guess}`")
                st.write(f"Sample values: {sample_values}")
                # Create a clean, unique options list
                all_options = [logic_guess, gpt_guess, "Time", "Numeric", "Ordinal", "Category", "ID", "Free Text"]
                # Remove duplicates while preserving order
                unique_options = []
                for option in all_options:
                    if option not in unique_options:
                        unique_options.append(option)
                
                user_type = st.selectbox(
                    f"Choose type for column '{col}'",
                    unique_options,
                    key=f"type_select_{col}"
                )
                user_choices.append({"column": col, "user_type": user_type})
            submitted = st.form_submit_button("Confirm Types")
            if submitted:
                st.session_state.user_confirmed_types = user_choices
                # Clear cleaning_result to force rerun
                st.session_state.cleaning_result = None
                st.rerun()
        st.stop()
    
    # --- User review and override of all column types ---
    if not mismatches:
        # Only show data type review if user clicks "Edit Data Types" button
        if st.session_state.get('show_type_review', False):
            st.subheader("Step 2: Review & Correct Column Data Types")
            logic_guesses = {}
            sample_values_dict = {}
            for col in df.columns:
                col_data = df[col]
                if pd.api.types.is_datetime64_any_dtype(col_data):
                    inferred = "Date"
                elif pd.api.types.is_numeric_dtype(col_data):
                    inferred = "Numeric"
                elif pd.api.types.is_categorical_dtype(col_data):
                    inferred = "Category"
                else:
                    inferred = "Text"
                logic_guesses[col] = inferred
                sample_values_dict[col] = col_data.dropna().astype(str).sample(min(5, len(col_data.dropna()))).tolist()
            user_types = {}
            warnings = []
            with st.form("column_type_review_form"):
                for col in df.columns:
                    inferred = logic_guesses.get(col, "Unknown")
                    sample = sample_values_dict.get(col, [])
                    warn = False
                    if inferred == "Numeric" and df[col].dropna().astype(str).str.match(r"^\d{5}$").mean() > 0.8:
                        warn = True
                        warnings.append(col)
                    # Fix duplicate options by using a clean options list
                    options = ["Numeric", "Category", "Text", "Date", "ID"]
                    default_idx = options.index(inferred) if inferred in options else 0
                    st.write(f"**{col}** (Sample: {sample})")
                    if warn:
                        st.warning(f"Column '{col}' looks like a Postal Code. Consider setting as 'Category'.")
                    user_type = st.selectbox(
                        f"Select data type for '{col}'",
                        options,
                        index=default_idx,
                        key=f"review_type_select_{col}"  # Unique key for each column
                    )
                    st.caption("ℹ️ Correct data types ensure meaningful charts and analysis. For example, treat Postal Codes as 'Category', not 'Numeric'.")
                    user_types[col] = user_type
                submitted = st.form_submit_button("Confirm Data Types")
                if submitted:
                    st.session_state['user_column_types'] = user_types
                    # Apply user types to DataFrame and update both JSON files
                    df = apply_user_types(df, user_types)
                    # Save to user_column_types.json (for analysis functions)
                    with open("user_column_types.json", "w") as f:
                        json.dump(user_types, f, indent=2)
                    # Save to final_column_types.json (for backward compatibility)
                    with open("final_column_types.json", "w") as f:
                        json.dump(user_types, f, indent=2)
                    st.success("Data types confirmed! You can now proceed to analytics below.")
                    st.session_state['show_analytics'] = True
                    st.session_state['show_type_review'] = False  # Hide the review form
                    st.rerun()
            
            # Cancel button to hide the review form
            if st.button("Cancel", key="cancel_type_review"):
                st.session_state['show_type_review'] = False
                st.rerun()
        else:
            # Show current data types first, then provide options
            st.subheader("Step 2: Data Type Management")
            
            # Show current inferred data types
            st.write("**Current Inferred Data Types:**")
            logic_guesses = {}
            
            # Use the original DataFrame for type inference to ensure all columns are included
            original_df = st.session_state.df
            
            # Improved column type inference with ID detection
            for col in original_df.columns:
                col_data = original_df[col]
                
                # Check for date columns first (highest priority)
                if pd.api.types.is_datetime64_any_dtype(col_data):
                    inferred = "Date"
                # Check for ID columns (but be more specific)
                elif any(id_keyword in col.lower() for id_keyword in ['id', 'row id', 'order id', 'customer id', 'product id', 'item id']):
                    # Only classify as ID if it's clearly an ID column and NOT a date column
                    if any(exact_id in col.lower() for exact_id in ['row id', 'order id', 'customer id', 'product id', 'item id']):
                        # Double check it's not a date column
                        if 'date' not in col.lower() and 'time' not in col.lower():
                            inferred = "ID"
                        else:
                            inferred = "Date"
                    else:
                        # For columns with 'id' but not exact matches, check if they're actually dates or other types
                        if pd.api.types.is_numeric_dtype(col_data):
                            inferred = "Numeric"
                        else:
                            inferred = "Text"
                elif pd.api.types.is_numeric_dtype(col_data):
                    inferred = "Numeric"
                elif pd.api.types.is_categorical_dtype(col_data):
                    inferred = "Category"
                else:
                    # For object/string columns, default to Text
                    inferred = "Text"
                
                logic_guesses[col] = inferred
            
            # Display current types in a nice format - show ALL columns
            type_display = {}
            for col, typ in logic_guesses.items():
                type_display[col] = typ
            
            st.json(type_display)
            
            # If user types are already set, show current types
            if 'user_column_types' in st.session_state:
                st.success("Data types have been configured. You can now proceed to the Model Pipeline to see analytics and train models.")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Edit Data Types", key="edit_types_main"):
                        st.session_state['show_type_review'] = True
                        st.rerun()
                with col2:
                    if st.button("Proceed with Current Types", key="proceed_current_types"):
                        # Ensure all required session state variables are set
                        st.session_state['show_analytics'] = True
                        st.session_state['confirmed'] = True  # Ensure confirmed is set
                        # Use current logic_guesses as user_column_types
                        st.session_state['user_column_types'] = logic_guesses
                        # Save current types to files for analysis functions
                        with open("user_column_types.json", "w") as f:
                            json.dump(logic_guesses, f, indent=2)
                        with open("final_column_types.json", "w") as f:
                            json.dump(logic_guesses, f, indent=2)
                        st.rerun()
            else:
                # First time - show options to use default types or edit
                st.info("Data types have been automatically inferred. You can proceed with these or edit them.")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Use Default Types & Proceed", key="use_default_types"):
                        st.session_state['user_column_types'] = logic_guesses
                        df = apply_user_types(df, logic_guesses)
                        # Save to user_column_types.json (for analysis functions)
                        with open("user_column_types.json", "w") as f:
                            json.dump(logic_guesses, f, indent=2)
                        # Save to final_column_types.json (for backward compatibility)
                        with open("final_column_types.json", "w") as f:
                            json.dump(logic_guesses, f, indent=2)
                        st.session_state['show_analytics'] = True
                        st.rerun()
                with col2:
                    if st.button("Edit Data Types", key="edit_types_default"):
                        st.session_state['show_type_review'] = True
                        st.rerun()



# --- Model Pipeline Section ---
if st.session_state.get('user_column_types') and st.session_state.get('confirmed') and st.session_state.get('df') is not None:
    user_types = st.session_state['user_column_types']
    df = apply_user_types(st.session_state.df.copy(), user_types)
    cleaned_csv_path = "data/cleaned_input.csv"
    df.to_csv(cleaned_csv_path, index=False)
    # Target column selection BEFORE pipeline run
    EXCLUDE_COLS = {'Unnamed: 0', 'id', 'ID', 'index'}
    # Exclude ID columns from numeric target selection
    id_columns = [col for col, typ in user_types.items() if typ.lower() == "id"]
    EXCLUDE_COLS.update(id_columns)
    
    numeric_columns = [col for col, typ in user_types.items() if typ.lower() == "numeric" and col.strip() not in EXCLUDE_COLS]
    if numeric_columns:
        st.subheader("Select Target Column for Model Training")
        target_col = st.selectbox("Select the numeric target column for model training", numeric_columns, key="target_col_select")
        st.session_state['target_col'] = target_col
        st.success(f"Target column selected: {target_col}")
        # Show Run Model Pipeline button only after target is selected
        if st.button("Run Model Pipeline", key="run_pipeline_main"):
            st.session_state['run_pipeline'] = True
            st.session_state['show_analytics'] = True
            st.rerun()
    else:
        st.error("No numeric columns available for target selection. Please ensure at least one column is set as 'Numeric'.")
        st.session_state['run_pipeline'] = False
        st.session_state['show_analytics'] = False
else:
    st.session_state['run_pipeline'] = False
    st.session_state['show_analytics'] = False

# --- Analytics and Pattern Analysis Section ---
if st.session_state.get('run_pipeline', False) and st.session_state.get('show_analytics', False):
    st.subheader("Model Pipeline")
    st.markdown("---")
    # Histogram configuration sidebar (only visible after pipeline is run)
    with st.sidebar:
        st.subheader("Histogram Settings")
        if 'num_bins' not in st.session_state:
            st.session_state.num_bins = 50
        if 'use_log_scale' not in st.session_state:
            st.session_state.use_log_scale = False
        if 'use_percentile_limit' not in st.session_state:
            st.session_state.use_percentile_limit = True
        num_bins = st.slider("Number of bins", min_value=10, max_value=100, value=st.session_state.num_bins, key="bin_slider")
        use_log_scale = st.checkbox("Use logarithmic scale", value=st.session_state.use_log_scale, key="log_scale_checkbox")
        use_percentile_limit = st.checkbox("Limit x-axis to 99th percentile", value=st.session_state.use_percentile_limit, key="percentile_checkbox")
        st.session_state.num_bins = num_bins
        st.session_state.use_log_scale = use_log_scale
        st.session_state.use_percentile_limit = use_percentile_limit
    # Run pattern analysis with user-confirmed types and selected target
    if st.session_state.get('target_col') is not None:
        with st.spinner("Running comprehensive pattern analysis..."):
            run_pattern_analysis_only(
                cleaned_csv_path=cleaned_csv_path,
                target_col=st.session_state['target_col'],
                num_bins=st.session_state.get('num_bins', 50),
                use_log_scale=st.session_state.get('use_log_scale', False),
                use_percentile_limit=st.session_state.get('use_percentile_limit', True)
            )
        st.success("Pattern analysis complete!")
        if os.path.exists("mansi/final_pattern_report.txt"):
            display_report("mansi/final_pattern_report.txt")
        # Start model training in background
        st.warning("Model training started in background... Please wait.")
        target_col_value = st.session_state.get('target_col')  # Get the value before starting thread
        def background_training():
            run_model_training_only(cleaned_csv_path=cleaned_csv_path, target_col=target_col_value)
            st.session_state["training_complete"] = True
        threading.Thread(target=background_training).start()
    else:
        st.error("Please select a target column before running the model pipeline.")
    if st.session_state.get("training_complete", False):
        st.success("Model training complete!")

