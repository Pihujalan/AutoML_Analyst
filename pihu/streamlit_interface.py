import streamlit as st
import pandas as pd
import os
def center(text, tag="p", size="16px"):
    st.markdown(
        f"<{tag} style='text-align: center; font-size:{size};'>{text}</{tag}>",
        unsafe_allow_html=True
    )

def left(text, tag="p", size="16px"):
    st.markdown(
        f"<{tag} style='text-align: left; font-size:{size};'>{text}</{tag}>",
        unsafe_allow_html=True
    )

def right(text, tag="p", size="16px"):
    st.markdown(
        f"<{tag} style='text-align: right; font-size:{size};'>{text}</{tag}>",
        unsafe_allow_html=True
    )

st.set_page_config(page_title="  AutoML  ", layout="wide")

center("AutoML Analyst ",tag="h1", size= "40px")
left("Upload your CSV dataset to begin.")
right("Upload a CSV file", tag="p",size="12px")
uploaded_file = st.file_uploader("", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Save uploaded file to /data/input.csv
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/input.csv", index=False)

    st.success("✅ File uploaded and saved successfully.")
    st.subheader("Preview of your data:")
    st.dataframe(df.head(10))

else:
    st.info("Please upload a CSV file to start.")
