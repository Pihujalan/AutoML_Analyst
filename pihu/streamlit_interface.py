import streamlit as st
import pandas as pd
import os

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
