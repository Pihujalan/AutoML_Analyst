import streamlit as st
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.fullmodel import run_pattern_analysis_only, run_model_training_only
import json
import threading
import base64

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
            df.to_csv("datasets/input.csv", index=False)

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
                    df.to_csv("datasets/input.csv", index=False)

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
    from data_processing.dataCleaning import clean_data_with_logs

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
    cleaned_path = "datasets/cleaned_input.csv"
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
    cleaned_csv_path = "datasets/cleaned_input.csv"
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
       

        import glob

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

        
def display_report(report_path):
    with open(report_path, "r") as file:
        content = file.read()

    st.subheader("📄 Business Insight Report")
    st.markdown(f"```markdown\n{content}\n```")

    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="final_pattern_report.txt">📥 Download Report as TXT</a>'
    st.markdown(href, unsafe_allow_html=True)

def background_training():
            run_model_training_only(cleaned_csv_path=cleaned_csv_path, target_col=target_col)
            st.session_state["training_complete"] = True  # flag will be checked by Streamlit later

        threading.Thread(target=background_training).start()
    

    # STEP 4: Notify user when training completes
    # My code searches for multiple file patterns:
    shap_files = [
        "random_forest_shap_importance.json",    # ✅ Handles rf_shap_importance.json
        "gbm_shap_importance.json",              # ✅ Handles gbm_shap_importance.json  
        "flaml_fallback_shap_importance.json",  # ✅ Handles flaml_fallback_shap_importance.json
        "shap_importance.json"                   # ✅ Generic fallback
    ]


    # ==================== ML PIPELINE RESULTS DISPLAY ====================
    # Add this section after the background training is started

    if st.session_state.get("training_complete", False):
        st.success("🎉 Model training complete!")
        st.markdown("---")
        st.subheader("📊 Machine Learning Pipeline Results")
        
        # 1. Model Evaluation Summary
        summary_files = ["final_model_summary.json", "summary.txt"]
        summary_found = False
        
        for summary_file in summary_files:
            if os.path.exists(summary_file):
                st.markdown("#### 📈 Model Evaluation Summary")
                try:
                    if summary_file.endswith('.json'):
                        with open(summary_file, "r") as f:
                            summary_data = json.load(f)
                        
                        # Display model type and metrics
                        st.write(f"**Model Used:** {summary_data.get('model_used', 'Unknown')}")
                        
                        # Display test metrics in a nice format
                        if 'test_metrics' in summary_data:
                            test_metrics = summary_data['test_metrics']
                            metrics_df = pd.DataFrame([{
                                'Metric': 'R² Score',
                                'Value': f"{test_metrics.get('r2', test_metrics.get('R2', 'N/A')):.4f}"
                            }, {
                                'Metric': 'MAE',
                                'Value': f"{test_metrics.get('mae', test_metrics.get('MAE', 'N/A')):.4f}"
                            }, {
                                'Metric': 'MSE', 
                                'Value': f"{test_metrics.get('mse', test_metrics.get('MSE', 'N/A')):.4f}"
                            }])
                            st.dataframe(metrics_df, use_container_width=True)
                        
                        # Display training metrics if available
                        if 'train_metrics' in summary_data:
                            st.write("**Training Performance:**")
                            train_metrics = summary_data['train_metrics']
                            st.write(f"- Training R²: {train_metrics.get('r2', 'N/A')}")
                            st.write(f"- Training MAE: {train_metrics.get('mae', 'N/A')}")
                            
                    else:  # .txt file
                        with open(summary_file, "r", encoding="utf-8") as f:
                            summary_content = f.read()
                        st.markdown(summary_content, unsafe_allow_html=True)
                        
                    summary_found = True
                    break
                    
                except Exception as e:
                    st.warning(f"Could not display summary from {summary_file}: {e}")
        
        if not summary_found:
            st.info("📊 Model evaluation summary not yet available...")

        # 2. SHAP Feature Importance
        shap_files = [
            "random_forest_shap_importance.json",
            "gbm_shap_importance.json", 
            "flaml_fallback_shap_importance.json",
            "shap_importance.json"
        ]
        
        shap_found = False
        for shap_file in shap_files:
            if os.path.exists(shap_file):
                st.markdown("#### 🌟 Top SHAP Feature Importances")
                try:
                    with open(shap_file, "r") as f:
                        shap_data = json.load(f)
                    
                    # Convert to DataFrame for better display
                    shap_df = pd.DataFrame([
                        {"Feature": feature, "SHAP Importance": f"{importance:.4f}"}
                        for feature, importance in shap_data.items()
                    ])
                    
                    st.dataframe(shap_df, use_container_width=True)
                    
                    # Show key insight
                    if shap_df.shape[0] > 0:
                        top_feature = shap_df.iloc[0]['Feature']
                        st.info(f"💡 **Key Insight:** '{top_feature}' has the highest impact on predictions")
                    
                    shap_found = True
                    break
                    
                except Exception as e:
                    st.warning(f"Could not display SHAP data from {shap_file}: {e}")
        
        if not shap_found:
            st.info("🌟 SHAP feature importance analysis not yet available...")

        # 3. Residual Analysis
        residual_files = ["residuals.csv", "residuals.txt"]
        residual_found = False
        
        for residual_file in residual_files:
            if os.path.exists(residual_file):
                st.markdown("#### 🔍 Residual Analysis")
                try:
                    if residual_file.endswith('.csv'):
                        residuals_df = pd.read_csv(residual_file)
                        
                        st.write("**Prediction vs Actual Sample:**")
                        st.dataframe(residuals_df.head(10), use_container_width=True)
                        
                        # Calculate and display residual statistics
                        if 'Residual' in residuals_df.columns:
                            residual_stats = {
                                'Mean Residual': f"{residuals_df['Residual'].mean():.4f}",
                                'Std Residual': f"{residuals_df['Residual'].std():.4f}",
                                'Min Residual': f"{residuals_df['Residual'].min():.4f}",
                                'Max Residual': f"{residuals_df['Residual'].max():.4f}"
                            }
                            
                            stats_df = pd.DataFrame([
                                {"Statistic": k, "Value": v} for k, v in residual_stats.items()
                            ])
                            st.dataframe(stats_df, use_container_width=True)
                    
                    else:  # .txt file
                        with open(residual_file, "r", encoding="utf-8") as f:
                            residual_content = f.read()
                        st.markdown(residual_content, unsafe_allow_html=True)
                    
                    residual_found = True
                    break
                    
                except Exception as e:
                    st.warning(f"Could not display residual analysis from {residual_file}: {e}")
        
        if not residual_found:
            st.info("🔍 Residual analysis not yet available...")

        # 4. Segment-wise Analysis
        segment_files = [
            "segment_analysis_random_forest.json",
            "segment_analysis_gbm.json",
            "segment_analysis_flaml_fallback.json", 
            "segment_analysis.json"
        ]
        
        segment_found = False
        for segment_file in segment_files:
            if os.path.exists(segment_file):
                st.markdown("#### 🔎 Segment-wise Performance Analysis")
                try:
                    with open(segment_file, "r") as f:
                        segment_data = json.load(f)
                    
                    # Display each segment category
                    for segment_col, results in segment_data.items():
                        st.write(f"**Segment by: {segment_col}**")
                        
                        if results and len(results) > 0:
                            # Convert to DataFrame for better display
                            segment_df = pd.DataFrame(results)
                            
                            # Format numeric columns to 4 decimal places
                            for col in segment_df.columns:
                                if segment_df[col].dtype in ['float64', 'float32']:
                                    segment_df[col] = segment_df[col].round(4)
                            
                            st.dataframe(segment_df, use_container_width=True)
                            
                            # Show best performing segment
                            if 'R2_Score' in segment_df.columns:
                                best_segment = segment_df.loc[segment_df['R2_Score'].idxmax()]
                                st.info(f"🏆 Best performing segment: {best_segment[segment_col]} (R² = {best_segment['R2_Score']:.4f})")
                        else:
                            st.write("No segment data available")
                        
                        st.markdown("---")
                    
                    segment_found = True
                    break
                    
                except Exception as e:
                    st.warning(f"Could not display segment analysis from {segment_file}: {e}")
        
        if not segment_found:
            st.info("🔎 Segment-wise analysis not yet available...")

        # 5. GPT Final Model Report with Download
        final_report_files = [
            "epic_final_report.txt",
            "gpt_final_report.txt", 
            "auto_training_report.txt"
        ]
        
        report_found = False
        for report_file in final_report_files:
            if os.path.exists(report_file):
                st.markdown("#### 🤖 AI-Generated Final Model Report")
                try:
                    with open(report_file, "r", encoding="utf-8") as f:
                        report_content = f.read()
                    
                    # Display the report
                    with st.container():
                        st.markdown(report_content, unsafe_allow_html=True)
                    
                    # Add download button (same style as pattern report)
                    b64 = base64.b64encode(report_content.encode()).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{report_file}">📥 Download Final Model Report</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    report_found = True
                    break
                    
                except Exception as e:
                    st.warning(f"Could not display final report from {report_file}: {e}")
        
        if not report_found:
            st.info("🤖 AI-generated final model report not yet available...")

    # Optional: Add a refresh button to check for new results
    if st.session_state.get("training_complete", False):
        if st.button("🔄 Refresh Results", help="Check for newly generated analysis files"):
            st.rerun()

    # Optional: Add a section to show which files are still being generated
    if not st.session_state.get("training_complete", False):
        st.info("🔄 Model training in progress... Results will appear here as they complete.")
        
        # Show progress of file generation
        expected_files = [
            "final_model_summary.json",
            "residuals.csv", 
            "epic_final_report.txt"
        ]
        
        progress_info = []
        for file in expected_files:
            if os.path.exists(file):
                progress_info.append(f"✅ {file}")
            else:
                progress_info.append(f"⏳ {file}")
        
        st.write("**File Generation Progress:**")
        for info in progress_info:
            st.write(info)
