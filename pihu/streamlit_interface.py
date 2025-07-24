import streamlit as st
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.fullmodel import run_pattern_analysis_only, run_model_training_only
import json
import threading
import base64
import glob


def display_report(report_path):
    with open(report_path, "r") as file:
        content = file.read()

    st.subheader("📄 Insight Report")
    with st.container():
        st.markdown(content, unsafe_allow_html=True)  
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="final_pattern_report.txt">📥 Download Report as TXT</a>'
    st.markdown(href, unsafe_allow_html=True)

# -------------------- Page Setup --------------------
st.set_page_config(page_title="AutoML Analyst", layout="wide")
st.title("AutoML Analyst")

# -------------------- Session State --------------------
if "confirmed" not in st.session_state:
    st.session_state.confirmed = False
if "df" not in st.session_state:
    st.session_state.df = None

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

    st.subheader("🧼 Cleaned Data Preview")
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from mansi.dataCleaning import clean_data_with_logs

    # --- Handle type mismatches interactively ---
    if "user_confirmed_types" not in st.session_state:
        st.session_state.user_confirmed_types = None
    if "cleaning_result" not in st.session_state:
        # First run: no user choices yet
        result = clean_data_with_logs(st.session_state.df, log_path="logs/cleaning_log.txt")
        if isinstance(result, tuple) and len(result) == 2:
            df, mismatches = result
            st.session_state.cleaning_result = (df, mismatches)
        else:
            st.session_state.cleaning_result = (result, [])
    # If user just submitted choices, rerun cleaning with those
    if st.session_state.user_confirmed_types is not None:
        cleaned_df = clean_data_with_logs(st.session_state.df, log_path="logs/cleaning_log.txt", user_confirmed_types=st.session_state.user_confirmed_types)
        st.session_state.cleaned_df = cleaned_df
        st.session_state.cleaning_result = (cleaned_df, [])
        st.session_state.user_confirmed_types = None
    df, mismatches = st.session_state.cleaning_result
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
                user_type = st.selectbox(
                    f"Choose type for column '{col}'",
                    [logic_guess, gpt_guess, "Time", "Numeric", "Ordinal", "Category", "ID", "Free Text"],
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
    # If no mismatches, show cleaned data and type summaries
    if isinstance(df, tuple):
        cleaned_df = df[0]
    else:
        cleaned_df = df
    st.session_state.cleaned_df = cleaned_df
    st.dataframe(cleaned_df.head(10))
    cleaned_path = "data/cleaned_input.csv"
    cleaned_df.to_csv(cleaned_path, index=False)
    with open(cleaned_path, "rb") as f:
        st.download_button(
            "📥 Download Cleaned CSV",
            f,
            file_name="cleaned_input.csv",
            mime="text/csv"
        )
    # Show type summaries
    st.subheader("Final Column Types")
    if os.path.exists("final_column_types.json"):
        with open("final_column_types.json") as f:
            final_types = json.load(f)
        st.json(final_types)
    st.subheader("User-Confirmed Column Types (if any)")
    if os.path.exists("user_column_types.json"):
        with open("user_column_types.json") as f:
            user_types = json.load(f)
        st.json(user_types)
    # Target selection and pipeline
    EXCLUDE_COLS = {'Unnamed: 0', 'id', 'ID', 'index'}

    numeric_columns = [
    col for col in cleaned_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if col.strip() not in EXCLUDE_COLS
]

    target_col = st.selectbox("Select the numeric target column for model training", numeric_columns)
    cleaned_csv_path = "data/cleaned_input.csv"
    cleaned_df.to_csv(cleaned_csv_path, index=False)
    print("Target column selected:", target_col)
   


    # Add this at the top if not already there
    if "training_complete" not in st.session_state:
        st.session_state["training_complete"] = False

    # Trigger pipeline
    if st.button("Run Full AutoML Pipeline 🚀"):
        
        # STEP 1: Run pattern analysis first
        with st.spinner("🔍 Running pattern analysis..."):
            run_pattern_analysis_only(cleaned_csv_path=cleaned_csv_path, target_col=target_col)
        st.success("✅ Pattern analysis complete!")
        display_report("mansi/final_pattern_report.txt")
        
        st.subheader("📊 Auto-Generated Pattern Visualizations")

        # Define plot categories and display logic
        plot_categories = {
            "Histograms": "histogram",
            "Boxplots": "boxplot",
            "Violin Plots": "violinplot",
            "Count Plots": "countplot",
            "Bar Plots": "barplot",
            "Time Series": "timeseries",
            "Scatter Plots": "scatterplot",
            "SHAP Summary": "shap",
            "Feature Importance": "feature_importance",
            "Correlation Heatmap": "correlation",
            "Conflicts": "conflict",
            "Model Test Evaluation": "test_"
        }

        # Dynamically loop and display plots
        for title, keyword in plot_categories.items():
            matching_files = glob.glob(f"plots/*{keyword}*.png")
            if matching_files:
                st.markdown(f"### 🔹 {title}")
                for file in matching_files:
                    st.image(file, use_column_width=True, caption=os.path.basename(file))
        # STEP 3: Start model training in background
        st.warning("⚙️ Model training started in background... Please wait.")

        def background_training():
            run_model_training_only(cleaned_csv_path=cleaned_csv_path, target_col=target_col)
            st.session_state["training_complete"] = True  # flag will be checked by Streamlit later

        threading.Thread(target=background_training).start()

    # STEP 4: Notify user when training completes
    if st.session_state.get("training_complete", False):
        st.success("✅ Model training complete!")