import pandas as pd
import numpy as np
import joblib
import json
import time
import warnings
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from flaml import AutoML
import os 
from category_encoders import TargetEncoder
from shap import TreeExplainer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
from scipy.stats import skew, kurtosis, zscore
import argparse
import sys
import re
from dateutil.parser import parse
from pandas.api.types import is_datetime64_any_dtype, is_object_dtype, is_string_dtype
import streamlit as st
import glob
import requests


# GPT report function using Together.ai
def call_llama3_business_report(prompt):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": "Bearer a81e0b6137ae03b2c1f53be56b1e1a99bfbe474de314a18694de143cc145a378",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "meta-llama/Llama-3-70b-chat-hf",
        "messages": [
            {
                "role": "system",
                "content": "You are a senior business data analyst. Write a full, in-depth business report based on data pattern analysis. Include tables, trend summaries, risks, recommendations, and graph interpretations."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.4,
        "max_tokens": 4096
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


#functions for explanation of graphs
def describe_numeric_col(df, col):
    mean = df[col].mean()
    std = df[col].std()
    skew = df[col].skew()
    min_val = df[col].min()
    max_val = df[col].max()

    skew_desc = (
        "right-skewed" if skew > 1 else
        "left-skewed" if skew < -1 else
        "approximately symmetric"
    )

    return (
        f"• '{col}' ranges from {min_val:.2f} to {max_val:.2f}, "
        f"mean: {mean:.2f}, std dev: {std:.2f}. "
        f"The distribution appears {skew_desc}."
    )

def describe_categorical_col(df, col):
    counts = df[col].value_counts(normalize=True).head(3)
    parts = [f"{idx} ({val:.1%})" for idx, val in counts.items()]
    return f"• Top categories in '{col}' are: " + ", ".join(parts) + "."

def describe_relationship(df, x_col, target_col):
    if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[target_col]):
        corr = df[[x_col, target_col]].corr().iloc[0, 1]
        direction = (
            "strong positive" if corr > 0.6 else
            "strong negative" if corr < -0.6 else
            "weak or no clear"
        )
        return f"• '{x_col}' vs '{target_col}' shows a {direction} correlation (r = {corr:.2f})."
    return None

def generate_pattern_graph_descriptions(df, target_col):
    summaries = []

    for col in df.columns:
        if col == target_col:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            summaries.append(describe_numeric_col(df, col))
            rel = describe_relationship(df, col, target_col)
            if rel:
                summaries.append(rel)

        elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
            summaries.append(describe_categorical_col(df, col))

    return "\n".join(summaries)

def sanitize_column_name(name):
    # Replace forbidden JSON characters for LightGBM with underscore
    name = re.sub(r'[\"\\\{\}\:\,\[\]]', '_', name)
    name = name.strip().replace(' ', '_')
    name = re.sub(r'_+', '_', name)
    return name


#====================================*Pattern Analysis*==============================================================
def run_pattern_analysis_only(cleaned_csv_path="data/cleaned_input.csv", target_col=None, num_bins=50, use_log_scale=False, use_percentile_limit=True):
    import pandas as pd
    import numpy as np
    import os
    import json
    import streamlit as st
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create 'plots' folder if it doesn't exist
    if not os.path.exists("plots"):
        os.mkdir("plots")

    # Check if CSV exists
    if not os.path.exists(cleaned_csv_path):
        st.error(f"❌ File not found: {cleaned_csv_path}")
        return
    
    # ✅ Clear old plots before creating new ones
    for old_plot in glob.glob("plots/*.png"):
        os.remove(old_plot)

    EXCLUDE_COLS = {'Unnamed: 0', 'id', 'ID', 'index'}

    # Load dataset
    df = pd.read_csv(cleaned_csv_path)

    # CRITICAL FIX: Always load and use the latest user-confirmed types, prioritizing user_column_types.json
    final_column_types = None
    try:
        with open("user_column_types.json", "r") as f:
            final_column_types = json.load(f)
            st.info("Loaded user-confirmed column types from user_column_types.json")
    except FileNotFoundError:
        try:
            with open("pihu/final_column_types.json", "r") as f:
                final_column_types = json.load(f)
                st.info("Loaded column types from pihu/final_column_types.json")
        except FileNotFoundError:
            try:
                with open("final_column_types.json", "r") as f:
                    final_column_types = json.load(f)
                    st.info("Loaded column types from final_column_types.json")
            except FileNotFoundError:
                try:
                    with open("../mansi/final_column_types.json", "r") as f:
                        final_column_types = json.load(f)
                        st.info("Loaded column types from ../mansi/final_column_types.json")
                except FileNotFoundError:
                    st.error("No data type configuration found. Please ensure column types are saved.")
                    return

    if not final_column_types:
        st.error("No valid column types configuration found.")
        return

    # Define force_convert_exact_dtypes function for this scope
    def force_convert_exact_dtypes(df, column_types):
        """
        General-purpose dtype converter using logical classification.

        Parameters:
        - df: pandas DataFrame (raw input)
        - column_types: dict with column_name → classification string
            valid types: 'numeric', 'category', 'ordinal', 'time', 'id'

        Returns:
        - Cleaned DataFrame with updated pandas dtypes
        """
        df_cleaned = df.copy()

        for col, raw_type in column_types.items():
            ctype = raw_type.strip().lower()

            try:
                # Fix string[python] which breaks category/numeric conversion
                if pd.api.types.is_string_dtype(df_cleaned[col]):
                    df_cleaned[col] = df_cleaned[col].astype("object")

                # Actual conversions
                if ctype == "time":
                    df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors="coerce")

                elif ctype == "numeric":
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce")
                    if (df_cleaned[col].dropna() % 1 == 0).all():
                        df_cleaned[col] = df_cleaned[col].astype("Int64")
                    else:
                        df_cleaned[col] = df_cleaned[col].astype("float64")

                elif ctype == "ordinal":
                    df_cleaned[col] = pd.Categorical(df_cleaned[col], ordered=True)

                elif ctype == "category":
                    df_cleaned[col] = df_cleaned[col].astype("category")

                elif ctype == "id":
                    df_cleaned[col] = df_cleaned[col].astype("object")

                elif ctype == "text":
                    df_cleaned[col] = df_cleaned[col].astype("object")

            except Exception as e:
                print(f"Failed to convert '{col}' to '{ctype}': {e}")

        return df_cleaned

    # CRITICAL FIX: Apply user-confirmed types BEFORE any analysis
    df = force_convert_exact_dtypes(df, final_column_types)
    
    # Replace bad strings globally AFTER dtype conversion
    df = df.replace(["Unknown", "N/A", "", "null", "None"], np.nan)

    # ✅ Preview
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    # 📊 Descriptive Statistics Section (only one heading)
    st.subheader("📊 Descriptive Statistics")
    
    # CRITICAL FIX: Build lists strictly by user-confirmed type for columns present in final_column_types
    numeric_columns = [col for col, typ in final_column_types.items() if typ.lower() == "numeric" and col in df.columns]
    category_columns = [col for col, typ in final_column_types.items() if typ.lower() == "category" and col in df.columns]
    text_columns = [col for col, typ in final_column_types.items() if typ.lower() == "text" and col in df.columns]
    date_columns = [col for col, typ in final_column_types.items() if typ.lower() in ["date", "datetime"] and col in df.columns]
    id_columns = [col for col, typ in final_column_types.items() if typ.lower() == "id" and col in df.columns]

    # For columns not present in final_column_types, fall back to pandas-inferred types
    all_user_cols = set(final_column_types.keys())
    remaining_numeric = [col for col in df.select_dtypes(include=[np.number]).columns if col not in all_user_cols]
    remaining_category = [col for col in df.select_dtypes(include=['category', 'object']).columns if col not in all_user_cols]
    remaining_text = [col for col in df.select_dtypes(include=['object', 'string']).columns if col not in all_user_cols]
    
    # Additional ID detection for columns not in user types
    remaining_ids = []
    for col in df.columns:
        if col not in all_user_cols:
            # Check if column name suggests it's an ID
            if any(id_keyword in col.lower() for id_keyword in ['id', 'row', 'order', 'customer', 'product', 'item']):
                remaining_ids.append(col)
    
    numeric_columns.extend(remaining_numeric)
    category_columns.extend(remaining_category)
    text_columns.extend(remaining_text)
    id_columns.extend(remaining_ids)

    # Remove any column from multiple lists (e.g., if user type is 'Category' but dtype is still numeric, only show in category)
    used_cols = set()
    def filter_unique(cols):
        out = []
        for c in cols:
            if c not in used_cols:
                out.append(c)
                used_cols.add(c)
        return out
    numeric_columns = filter_unique(numeric_columns)
    category_columns = filter_unique(category_columns)
    text_columns = filter_unique(text_columns)
    id_columns = filter_unique(id_columns)

    # CRITICAL FIX: Ensure we only use the correctly filtered columns for analysis
    # Filter based on user-confirmed types, NOT pandas dtypes
    numeric_columns = [col for col in numeric_columns if col in final_column_types and final_column_types[col].lower() == "numeric"]
    category_columns = [col for col in category_columns if col in final_column_types and final_column_types[col].lower() == "category"]
    text_columns = [col for col in text_columns if col in final_column_types and final_column_types[col].lower() == "text"]
    date_columns = [col for col in date_columns if col in final_column_types and final_column_types[col].lower() in ["date", "datetime"]]
    id_columns = [col for col in id_columns if col in final_column_types and final_column_types[col].lower() == "id"]

    # Numeric stats
    if numeric_columns:
        numeric_stats = df[numeric_columns].describe().T.fillna("-")
        st.markdown("#### 📐 Numeric Columns")
        st.dataframe(numeric_stats, use_container_width=True)
    else:
        st.info("No numeric columns found.")
        
    # Categorical stats
    if category_columns:
        try:
            cat_stats = df[category_columns].describe(include='all').T.fillna("-")
            st.markdown("#### 📊 Categorical Columns")
            st.dataframe(cat_stats, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate categorical stats: {e}")
    else:
        st.info("No categorical columns found.")
        
    # Text stats
    if text_columns:
        try:
            text_stats = df[text_columns].describe(include='all').T.fillna("-")
            st.markdown("#### 📝 Text Columns")
            st.dataframe(text_stats, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate text stats: {e}")
    else:
        st.info("No text columns found.")

    # Date stats
    if date_columns:
        try:
            date_stats = df[date_columns].describe().T.fillna("-")
            st.markdown("#### 📅 Date Columns")
            st.dataframe(date_stats, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate date stats: {e}")
    else:
        st.info("No date columns found.")

    # ID stats
    if id_columns:
        try:
            id_stats = df[id_columns].describe().T.fillna("-")
            st.markdown("#### 🔍 ID Columns")
            st.dataframe(id_stats, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate ID stats: {e}")
    else:
        st.info("No ID columns found.")

    # Display column type summary
    st.markdown("#### 🔍 Column Type Summary")
    type_summary = []
    for col in df.columns:
        if col in final_column_types:
            type_summary.append(f"**{col}**: {final_column_types[col]} (user-confirmed)")
        else:
            type_summary.append(f"**{col}**: {df[col].dtype} (auto-inferred)")
    
    st.markdown("\n".join(type_summary))

    # ==================== VISUALIZATION SECTION ====================
    st.markdown("---")
    st.subheader("📊 Data Visualizations")
    
    # 1. NUMERIC COLUMNS - HISTOGRAMS
    if numeric_columns:
        st.markdown("#### 📈 Histograms for Numeric Columns")
        for col in numeric_columns:
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Apply histogram settings
                if use_percentile_limit:
                    # Calculate 99th percentile limit
                    percentile_99 = df[col].quantile(0.99)
                    plot_data = df[col][df[col] <= percentile_99]
                else:
                    plot_data = df[col]
                
                # Create histogram
                if use_log_scale:
                    # Use log scale for y-axis
                    plt.hist(plot_data.dropna(), bins=num_bins, alpha=0.7, edgecolor='black')
                    plt.yscale('log')
                else:
                    plt.hist(plot_data.dropna(), bins=num_bins, alpha=0.7, edgecolor='black')
                
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.warning(f"Could not create histogram for {col}: {e}")
    
    # 2. CATEGORICAL COLUMNS - BAR CHARTS
    if category_columns:
        st.markdown("#### 📊 Bar Charts for Categorical Columns")
        for col in category_columns:
            try:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Get value counts and limit to top 20 categories for readability
                value_counts = df[col].value_counts().head(20)
                
                plt.bar(range(len(value_counts)), value_counts.values, alpha=0.7, edgecolor='black')
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.warning(f"Could not create bar chart for {col}: {e}")
    
    # 3. CORRELATION HEATMAP (NUMERIC COLUMNS ONLY)
    if len(numeric_columns) > 1:
        st.markdown("#### 🔗 Correlation Heatmap (Numeric Columns)")
        try:
            # Create correlation matrix for numeric columns only
            corr_matrix = df[numeric_columns].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=0.5, cbar_kws={"shrink": .8})
            plt.title('Correlation Matrix of Numeric Variables')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        except Exception as e:
            st.warning(f"Could not create correlation heatmap: {e}")
    else:
        st.info(f"Correlation heatmap requires at least 2 numeric columns. Found: {len(numeric_columns)}")
    
    # 4. BOX PLOTS FOR NUMERIC COLUMNS
    if numeric_columns:
        st.markdown("#### 📦 Box Plots for Numeric Columns")
        for col in numeric_columns:
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Apply percentile limit if enabled
                if use_percentile_limit:
                    percentile_99 = df[col].quantile(0.99)
                    plot_data = df[col][df[col] <= percentile_99]
                else:
                    plot_data = df[col]
                
                plt.boxplot(plot_data.dropna())
                plt.title(f'Box Plot of {col}')
                plt.ylabel(col)
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.warning(f"Could not create box plot for {col}: {e}")
    
    # 5. SCATTER PLOTS (if target column is numeric)
    if target_col and target_col in numeric_columns and len(numeric_columns) > 1:
        st.markdown("#### 🎯 Scatter Plots vs Target Variable")
        other_numeric = [col for col in numeric_columns if col != target_col]
        
        for col in other_numeric[:5]:  # Limit to first 5 to avoid too many plots
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Apply percentile limit if enabled
                if use_percentile_limit:
                    percentile_99_x = df[col].quantile(0.99)
                    percentile_99_y = df[target_col].quantile(0.99)
                    plot_data = df[(df[col] <= percentile_99_x) & (df[target_col] <= percentile_99_y)]
                else:
                    plot_data = df
                
                plt.scatter(plot_data[col], plot_data[target_col], alpha=0.5)
                plt.xlabel(col)
                plt.ylabel(target_col)
                plt.title(f'{col} vs {target_col}')
                plt.grid(True, alpha=0.3)
                
                # Add trend line
                z = np.polyfit(plot_data[col].dropna(), plot_data[target_col].dropna(), 1)
                p = np.poly1d(z)
                plt.plot(plot_data[col], p(plot_data[col]), "r--", alpha=0.8)
                
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.warning(f"Could not create scatter plot for {col} vs {target_col}: {e}")
    
    # 6. TEXT COLUMNS - WORD FREQUENCY (if applicable)
    if text_columns:
        st.markdown("#### 📝 Text Column Analysis")
        for col in text_columns[:3]:  # Limit to first 3 text columns
            try:
                # Show sample of unique values
                unique_values = df[col].value_counts().head(10)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                plt.bar(range(len(unique_values)), unique_values.values, alpha=0.7, edgecolor='black')
                plt.title(f'Top 10 Most Common Values in {col}')
                plt.xlabel('Values')
                plt.ylabel('Count')
                plt.xticks(range(len(unique_values)), [str(x)[:20] + '...' if len(str(x)) > 20 else str(x) for x in unique_values.index], 
                          rotation=45, ha='right')
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.warning(f"Could not create text analysis for {col}: {e}")

    # 7. DATE COLUMNS - TIME SERIES PLOTS
    if date_columns:
        st.markdown("#### 📅 Time Series Analysis")
        for col in date_columns:
            try:
                # Create time series plot
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Group by date and count occurrences
                date_counts = df[col].value_counts().sort_index()
                
                plt.plot(date_counts.index, date_counts.values, marker='o', linewidth=2, markersize=4)
                plt.title(f'Time Series: {col}')
                plt.xlabel('Date')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.warning(f"Could not create time series plot for {col}: {e}")
    
    # 8. OUTLIER DETECTION FOR NUMERIC COLUMNS
    if numeric_columns:
        st.markdown("#### 🔍 Outlier Analysis")
        for col in numeric_columns:
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Z-score method
                z_scores = np.abs(zscore(df[col].dropna()))
                outliers_z = z_scores > 3
                
                # IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers_iqr = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
                
                # Plot Z-score outliers
                ax1.scatter(range(len(df[col])), df[col], c=outliers_z, cmap='viridis', alpha=0.6)
                ax1.set_title(f'Z-Score Outliers in {col}')
                ax1.set_xlabel('Index')
                ax1.set_ylabel(col)
                
                # Plot IQR outliers
                ax2.scatter(range(len(df[col])), df[col], c=outliers_iqr, cmap='viridis', alpha=0.6)
                ax2.set_title(f'IQR Outliers in {col}')
                ax2.set_xlabel('Index')
                ax2.set_ylabel(col)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Show outlier statistics
                outlier_stats = {
                    'Z-Score Outliers': outliers_z.sum(),
                    'IQR Outliers': outliers_iqr.sum(),
                    'Total Records': len(df[col].dropna())
                }
                st.write(f"**{col} Outlier Statistics:**")
                st.write(outlier_stats)
                
            except Exception as e:
                st.warning(f"Could not create outlier analysis for {col}: {e}")

    # ==================== PATTERN REPORT GENERATION ====================
    # Generate fresh pattern descriptions with current dataset's column names
    if target_col and target_col in df.columns:
        try:
            # Generate pattern graph descriptions with current dataset
            pattern_descriptions = generate_pattern_graph_descriptions(df, target_col)
            
            # Save to graph_descriptions.txt
            with open("mansi/graph_descriptions.txt", "w") as f:
                f.write(pattern_descriptions)
            
            # Generate comprehensive business report
            # Create a summary of the analysis for the business report
            analysis_summary = {
                "dataset_shape": df.shape,
                "numeric_columns": numeric_columns,
                "categorical_columns": category_columns,
                "text_columns": text_columns,
                "date_columns": date_columns,
                "id_columns": id_columns,
                "target_column": target_col,
                "pattern_descriptions": pattern_descriptions
            }
            
            # Create business report prompt
            business_prompt = f"""
            Based on the following dataset analysis, create a comprehensive business report:
            
            Dataset Information:
            - Shape: {df.shape}
            - Target Column: {target_col}
            
            Column Types:
            - Numeric Columns: {numeric_columns}
            - Categorical Columns: {category_columns}
            - Text Columns: {text_columns}
            - Date Columns: {date_columns}
            - ID Columns: {id_columns}
            
            Pattern Analysis:
            {pattern_descriptions}
            
            Please create a comprehensive business report with:
            1. Executive Summary
            2. Data Quality & Business Risk Assessment
            3. Insights from Visualizations & Stats
            4. Correlations & Associations
            5. Statistical Test Results
            6. Segment-Level Observations
            7. Time Trends
            8. Business Risks & Data Limitations
            9. Strategic Recommendations
            10. Visual Dashboards & Monitoring Suggestions
            
            Use the actual column names from the dataset in all analysis and recommendations.
            """
            
            # Generate business report using the existing function
            business_report = call_llama3_business_report(business_prompt)
            
            # Save to final_pattern_report.txt
            with open("mansi/final_pattern_report.txt", "w") as f:
                f.write(business_report)
            
            st.success("✅ Pattern analysis reports generated with current dataset column names!")
            
        except Exception as e:
            st.warning(f"Could not generate pattern reports: {e}")

    # Continue with the rest of the analysis using the correctly filtered columns...


def run_model_training_only(cleaned_csv_path="data/cleaned_input.csv", target_col=None):
        # Define force_convert_exact_dtypes function for this scope
        def force_convert_exact_dtypes(df, column_types):
            """
            General-purpose dtype converter using logical classification.

            Parameters:
            - df: pandas DataFrame (raw input)
            - column_types: dict with column_name → classification string
                valid types: 'numeric', 'category', 'ordinal', 'time', 'id'

            Returns:
            - Cleaned DataFrame with updated pandas dtypes
            """
            df_cleaned = df.copy()

            for col, raw_type in column_types.items():
                ctype = raw_type.strip().lower()

                try:
                    # Fix string[python] which breaks category/numeric conversion
                    if pd.api.types.is_string_dtype(df_cleaned[col]):
                        df_cleaned[col] = df_cleaned[col].astype("object")

                    # Actual conversions
                    if ctype == "time":
                        df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors="coerce")

                    elif ctype == "numeric":
                        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce")
                        if (df_cleaned[col].dropna() % 1 == 0).all():
                            df_cleaned[col] = df_cleaned[col].astype("Int64")
                        else:
                            df_cleaned[col] = df_cleaned[col].astype("float64")

                    elif ctype == "ordinal":
                        df_cleaned[col] = pd.Categorical(df_cleaned[col], ordered=True)

                    elif ctype == "category":
                        df_cleaned[col] = df_cleaned[col].astype("category")

                    elif ctype == "id":
                        df_cleaned[col] = df_cleaned[col].astype("object")

                    elif ctype == "text":
                        df_cleaned[col] = df_cleaned[col].astype("object")

                except Exception as e:
                    print(f"Failed to convert '{col}' to '{ctype}': {e}")

            return df_cleaned

        df = pd.read_csv(cleaned_csv_path)
        print("Original shape:", df.shape)

        # CRITICAL FIX: Always load and use the latest user-confirmed types, prioritizing user_column_types.json
        final_column_types = None
        try:
            with open("user_column_types.json", "r") as f:
                final_column_types = json.load(f)
                print("Loaded user-confirmed column types from user_column_types.json")
        except FileNotFoundError:
            try:
                with open("pihu/final_column_types.json", "r") as f:
                    final_column_types = json.load(f)
                    print("Loaded column types from pihu/final_column_types.json")
            except FileNotFoundError:
                try:
                    with open("final_column_types.json", "r") as f:
                        final_column_types = json.load(f)
                        print("Loaded column types from final_column_types.json")
                except FileNotFoundError:
                    try:
                        with open("../mansi/final_column_types.json", "r") as f:
                            final_column_types = json.load(f)
                            print("Loaded column types from ../mansi/final_column_types.json")
                    except FileNotFoundError:
                        print("No data type configuration found. Please ensure column types are saved.")
                        return

        if not final_column_types:
            print("No valid column types configuration found.")
            return

        # CRITICAL FIX: Apply user-confirmed types BEFORE any model training
        df = force_convert_exact_dtypes(df, final_column_types)

        # Target column input (only used for Target Encoding, not applied here)
        if target_col not in df.columns:
            raise ValueError(f"❌ Target column '{target_col}' not found in the dataset.")

        # Save the target column config
        with open("mansi/target_config.json", "w") as f:
            json.dump({"target_column": target_col}, f)

        # --- STEP 1: Convert boolean-like columns to 0/1 ---
        for col in df.columns:
            unique_vals = set(df[col].dropna().unique())
            if unique_vals <= {True, False} or unique_vals <= {'True', 'False'}:
                df[col] = df[col].replace({True: 1, False: 0, 'True': 1, 'False': 0}).astype(int)

        # --- STEP 2: Manually normalize key binary columns ---
        if 'gender' in df.columns:
            df['gender'] = df['gender'].astype(str).str.lower().str.strip()
            df['gender'] = df['gender'].replace({
                'male': 1, 'm': 1,
                'female': 0, 'f': 0
            })

        if 'everbenched' in df.columns:
            df['everbenched'] = df['everbenched'].astype(str).str.lower().str.strip()
            df['everbenched'] = df['everbenched'].replace({
                'yes': 1, 'y': 1,
                'no': 0, 'n': 0
            })

        # --- STEP 3: Label encode remaining object/category columns ---
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cat_cols:
            df[col] = df[col].astype(str).str.lower().str.strip()  # Clean text
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            cwd = os.getcwd()
            encoders_dir = os.path.join(cwd, "mansi", "encoders")
            os.makedirs(encoders_dir, exist_ok=True)
            joblib.dump(le, os.path.join(encoders_dir, f"{col}_encoder.pkl"))

        # --- STEP 4: Save final encoded data ---
        df.to_csv("mansi/encoded_input.csv", index=False)
        print("\n✅ Encoding complete. Output saved as 'encoded_input.csv'")
        print("🧠 LabelEncoders saved in 'encoders/' folder.")

        #========================train-test-split=======================================
        df = pd.read_csv("mansi/encoded_input.csv")
        df.columns = [sanitize_column_name(col) for col in df.columns]

        # Load target column from earlier saved config
        with open("mansi/target_config.json", "r") as f:
            target_col = json.load(f)["target_column"]

        if target_col not in df.columns:
            print(f"❌ Target column '{target_col}' not found in the dataset. Exiting.")
            exit()

        # CRITICAL FIX: Use the same final_column_types that was loaded earlier
        # Don't reload from file again, use the one already loaded and applied
        cols_to_exclude = [col for col, typ in final_column_types.items() if typ.lower() in ["id", "datetime", "time"]]
    

        X = df.drop(columns=cols_to_exclude + [target_col], errors='ignore')


        Y = df[target_col]
        print(f" Features and target separated. Shape of X: {X.shape}, Y: {Y.shape}")




        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        X_train.to_csv("mansi/X_train.csv", index=False)
        X_test.to_csv("mansi/X_test.csv", index=False)
        Y_train.to_csv("mansi/Y_train.csv", index=False)
        Y_test.to_csv("mansi/Y_test.csv", index=False)



        print("\n✅ Data split completed:")
        print(f"   - X_train: {X_train.shape}")
        print(f"   - X_test: {X_test.shape}")
        print(f"   - Y_train: {Y_train.shape}")
        print(f"   - Y_test: {Y_test.shape}")

        def looks_like_dt(val):
            try:
                parse(val)
                return True
            except Exception:
                return False

        def looks_like_date(series):
            sample = series.dropna().astype(str).head(10)
            for x in sample:
                if (re.match(r"\d{4}-\d{2}-\d{2}", x) or
                    re.match(r"\d{2,4}/\d{1,2}/\d{1,2}", x) or
                    looks_like_dt(x)):
                    return True
            return False

        date_cols = [
            col for col in X_train.columns
            if (is_object_dtype(X_train[col]) or is_datetime64_any_dtype(X_train[col]))
            and (looks_like_date(X_train[col]) or
                    "date" in col.lower() or "time" in col.lower() or "year" in col.lower())
        ]

        for col in date_cols:
            for df in [X_train, X_test]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    df[f"{col}_year"] = df[col].dt.year
                    df[f"{col}_month"] = df[col].dt.month
                    df[f"{col}_day"] = df[col].dt.day
                    df[f"{col}_dow"] = df[col].dt.dayofweek
                    df.drop(columns=[col], inplace=True)

            # List or auto-detect your date columns here
        DATE_COLS = [col for col in X_train.columns if any(word in col.lower() for word in ["date", "year", "dob", "doj", "joined", "created", "timestamp"])]

        for col in DATE_COLS:
            for df in [X_train, X_test]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    df[f"{col}_year"] = df[col].dt.year
                    df[f"{col}_month"] = df[col].dt.month
                    df[f"{col}_day"] = df[col].dt.day
                    df[f"{col}_dow"] = df[col].dt.dayofweek
                    df.drop(columns=[col], inplace=True)


        #--------------------------------GBM-----------------------------------------------------
        to_drop = [col for col in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[col])]
        if to_drop:
            print(f"Dropping non-numeric columns: {to_drop}")
            X_train = X_train.drop(columns=to_drop)
            X_test = X_test.drop(columns=to_drop)

        X_train.columns = [sanitize_column_name(col) for col in X_train.columns]
        X_test.columns  = [sanitize_column_name(col) for col in X_test.columns]

        non_num_cols = [col for col in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[col])]
        print("Non-numeric columns about to be passed to modeling:", non_num_cols)
        for col in non_num_cols:
            print(X_train[col].head())
        # Drop EVERY non-numeric column (dates, strings, etc)
        if non_num_cols:
            X_train = X_train.drop(columns=non_num_cols)
            X_test = X_test.drop(columns=[c for c in non_num_cols if c in X_test.columns])


        def train_gbm_model(X_train, y_train, X_test, y_test, target_col):
            print("🚀 Training GBM Model...")
            baseline_model = GradientBoostingRegressor(random_state=42)
            baseline_model.fit(X_train, y_train)
            y_pred_baseline = baseline_model.predict(X_test)

            mae_base = mean_absolute_error(y_test, y_pred_baseline)
            mse_base = mean_squared_error(y_test, y_pred_baseline)
            r2_base = r2_score(y_test, y_pred_baseline)
            print(f"📊 Baseline Performance:\nMAE: {mae_base:.2f} | MSE: {mse_base:.2f} | R²: {r2_base:.2f}")

            # ------------------ Hyperparameter Tuning ------------------
            print("\n🔍 Running Randomized Search...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 4, 5],
                'subsample': [0.6, 0.8, 1.0],
                'min_samples_split': [2, 5, 10]
            }

            rsearch = RandomizedSearchCV(
                estimator=GradientBoostingRegressor(random_state=42),
                param_distributions=param_grid,
                n_iter=10,
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1,
                random_state=42
            )

            rsearch.fit(X_train, y_train)
            best_model = rsearch.best_estimator_
            print("✅ Best parameters found:", rsearch.best_params_)

            # ------------------ Evaluate Tuned Model ------------------
            print("📌 Tuned Performance:")
            y_pred = best_model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"MAE: {mae:.2f} | MSE: {mse:.2f} | R²: {r2:.2f}")

            joblib.dump(best_model, "trained_gbm_model_tuned.pkl")
                
            train_preds = best_model.predict(X_train)
            r2_train = r2_score(y_train, train_preds)
            gap = round(r2_train - r2, 4)
            print(f"R² (Train): {r2_train:.2f}, R² (Test): {r2:.2f}, Gap: {gap:.4f}")

            X_train.columns = [sanitize_column_name(col) for col in X_train.columns]
            X_test.columns  = [sanitize_column_name(col) for col in X_test.columns]

            non_num_cols = [col for col in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[col])]
            print("Non-numeric columns about to be passed to modeling:", non_num_cols)
            for col in non_num_cols:
                print(X_train[col].head())
            # Drop EVERY non-numeric column (dates, strings, etc)
            if non_num_cols:
                X_train = X_train.drop(columns=non_num_cols)
                X_test = X_test.drop(columns=[c for c in non_num_cols if c in X_test.columns])


            if r2 < 0.5 or (gap > 0.15 and r2 < 0.5):
                print(f"\n⚠️ GBM model underperformed or overfitting detected (gap = {gap})")

                automl = AutoML()
                automl.fit(X_train=X_train, y_train=y_train, task="regression",estimator_list=['lgbm', 'xgboost'], time_budget=60,n_jobs=-1)

                y_pred_fallback = automl.predict(X_test)
                mae_fallback = mean_absolute_error(y_test, y_pred_fallback)
                mse_fallback = mean_squared_error(y_test, y_pred_fallback)
                r2_fallback = r2_score(y_test, y_pred_fallback)

                final_model = automl
                used_model_type = "flaml_fallback"

                print("\n✅ Fallback model trained and selected.")
                print(f"FLAML Performance:\nMAE: {mae_fallback:.2f} | MSE: {mse_fallback:.2f} | R²: {r2_fallback:.2f}")
            else:
                final_model = best_model
                used_model_type = "gbm"
                print("✅ GBM model performed well. No fallback needed.")

            # ------------------ Save Final Model ------------------
            joblib.dump(final_model, "final_model.pkl")
            with open("model_type.txt", "w") as f:
                f.write(used_model_type)
            test_metrics = {
                "MAE": mae if used_model_type == "gbm" else mae_fallback,
                "MSE": mse if used_model_type == "gbm" else mse_fallback,
                "R2": r2 if used_model_type == "gbm" else r2_fallback
            }
            print("\n📊 Final Model Test Performance:")
            print(f"Model Type Used: {used_model_type.upper()}")
            print(f"MAE: {test_metrics['MAE']:.2f}")
            print(f"MSE: {test_metrics['MSE']:.2f}")
            print(f"R²: {test_metrics['R2']:.2f}")
            return final_model, used_model_type, test_metrics


        #-----------------random forest----------------------------------------------------------
        X_train.columns = [sanitize_column_name(col) for col in X_train.columns]
        X_test.columns  = [sanitize_column_name(col) for col in X_test.columns]
        non_num_cols = [col for col in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[col])]
        print("Non-numeric columns about to be passed to modeling:", non_num_cols)
        for col in non_num_cols:
            print(X_train[col].head())
        # Drop EVERY non-numeric column (dates, strings, etc)
        if non_num_cols:
            X_train = X_train.drop(columns=non_num_cols)
            X_test = X_test.drop(columns=[c for c in non_num_cols if c in X_test.columns])

        def train_rf_model(X_train, y_train, X_test, y_test):
            num_rows, num_cols = X_train.shape
            target_std = y_train.std()
            target_uniques = y_train.nunique()

            print(f"\n📊 Data Profile:")
            print(f"   - Rows: {num_rows}")
            print(f"   - Columns: {num_cols}")
            print(f"   - Target Std Dev: {round(target_std, 2)}")
            print(f"   - Unique Target Values: {target_uniques}")

            if num_rows < 300:
                cv = 3
            elif num_rows < 1000:
                cv = 5
            else:
                cv = 7

            n_estimators_range = [300, 600, 1000] if target_std > 5 else [100, 300, 500]
            max_depth_range = [30, 50, None] if target_uniques > 20 else [10, 20, 30]
            min_samples_leaf_range = [1, 2] if num_rows > 1000 else [2, 3]

            param_grid = {
                'n_estimators': n_estimators_range,
                'max_depth': max_depth_range,
                'min_samples_split': [2],
                'min_samples_leaf': min_samples_leaf_range,
                'max_features': ['sqrt', 'log2'],
                'bootstrap': [True]
            }

            total_combinations = (
                len(param_grid['n_estimators']) *
                len(param_grid['max_depth']) *
                len(param_grid['min_samples_split']) *
                len(param_grid['min_samples_leaf']) *
                len(param_grid['max_features']) *
                len(param_grid['bootstrap'])
            )
            n_iter = min(50, total_combinations)

            print(f"\n🧪 Search Space: {total_combinations} combos | n_iter = {n_iter} | cv = {cv}")

            rf = RandomForestRegressor(random_state=42)

            search = RandomizedSearchCV(
                estimator=rf,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                verbose=2,
                n_jobs=-1,
                scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'],
                refit='r2',
                random_state=42
            )

            search.fit(X_train, y_train)
            print("✅ FIT done")

            best_rf = search.best_estimator_
        
            train_preds = best_rf.predict(X_train)
            joblib.dump(best_rf, "best_rf_model.pkl")

            
            r2 = r2_score(y_train, train_preds)
            mae = mean_absolute_error(y_train, train_preds)
            mse = mean_squared_error(y_train, train_preds)
            rmse = np.sqrt(mse)


            print("\nBest Hyperparameters Found:")
            print(search.best_params_)

            print("\n📊 Model Performance on Training Data:")
            print(f"   - R² Score:  {round(r2, 4)}")
            print(f"   - MAE:       {round(mae, 4)}")
            print(f"   - RMSE:      {round(rmse, 4)}")

            with open("best_rf_metrics.json", "w") as f:
                json.dump({
                    "r2": round(r2, 4),
                    "mae": round(mae, 4),
                    "rmse": round(rmse, 4),
                    "cv": cv,
                    "n_iter": n_iter,
                    "param_grid_used": param_grid,
                    "best_params": search.best_params_,
                    "data_profile": {
                        "rows": num_rows,
                        "columns": num_cols,
                        "target_std": round(target_std, 2),
                        "target_uniques": target_uniques
                    }
                    }, f, indent=4)
            # ------------------ Segment-wise Analysis ------------------
        #both at the end of file
        #--------------------end of feture analysis-------------------------------------------------------
            joblib.dump(best_rf, "best_rf_model.pkl")

            with open("best_rf_metrics.json", "w") as f:
                json.dump({
                    "r2": round(r2, 4),
                    "mae": round(mae, 4),
                    "rmse": round(rmse, 4),
                    "cv": cv,
                    "n_iter": n_iter,
                    "param_grid_used": param_grid,
                    "best_params": search.best_params_,
                    "data_profile": {
                        "rows": num_rows,
                        "columns": num_cols,
                        "target_std": round(target_std, 2),
                        "target_uniques": target_uniques
                    }
                }, f, indent=4)

            test_preds = best_rf.predict(X_test)
            test_r2 = r2_score(y_test, test_preds)
            test_mae = mean_absolute_error(y_test, test_preds)
            test_mse = mean_squared_error(y_test, test_preds)
            test_rmse = np.sqrt(test_mse)


            print("\n📊 Test Set Performance:")
            print(f"   - R² Score:  {round(test_r2, 4)}")
            print(f"   - MAE:       {round(test_mae, 4)}")
            print(f"   - RMSE:      {round(test_rmse, 4)}")

            
            gap = round(r2 - test_r2, 4)
            if gap > 0.15 and test_r2 < 0.5:
                print("\n⚠️ Likely Overfitting Detected!")
                print(f"   - Train R² is {gap} higher than Test R²")
            elif gap < 0.05:
                print("\n✅ Model is well-generalized!")
            else:
                print("\nℹ️ Slight overfitting, may be acceptable depending on use case.")

            return best_rf, {"r2": r2, "mae": mae, "rmse": rmse}, {"r2": test_r2, "mae": test_mae, "rmse": test_rmse}

        # 🔽 🔽 🔽 FUNCTION CALL — DO NOT REMOVE 🔽 🔽 🔽
        X_train = pd.read_csv("mansi/X_train.csv")
        y_train = pd.read_csv("mansi/Y_train.csv").squeeze()
        X_test = pd.read_csv("mansi/X_test.csv")
        y_test = pd.read_csv("mansi/Y_test.csv").squeeze()
        X_train.columns = [sanitize_column_name(col) for col in X_train.columns]
        X_test.columns  = [sanitize_column_name(col) for col in X_test.columns]
        non_num_cols = [col for col in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[col])]
        print("Non-numeric columns about to be passed to modeling:", non_num_cols)
        for col in non_num_cols:
            print(X_train[col].head())
        # Drop EVERY non-numeric column (dates, strings, etc)
        if non_num_cols:
            X_train = X_train.drop(columns=non_num_cols)
            X_test = X_test.drop(columns=[c for c in non_num_cols if c in X_test.columns])

        

        def clean_target(y):
            # If Series is object or datetime, handle conversion...
            if isinstance(y, pd.DataFrame):
                y = y.squeeze()
            if pd.api.types.is_numeric_dtype(y):
                return y
            # Try numeric conversion first
            try:
                y_num = pd.to_numeric(y, errors="raise")
                return y_num
            except Exception:
                pass
            # Try datetime conversion next
            try:
                y_date = pd.to_datetime(y, errors="raise")
                # Convert to days since epoch, as float
                y_num = (y_date - pd.Timestamp("1970-01-01")).dt.days.astype(float)
                print("Converted datetime target to days since 1970-01-01.")
                return y_num
            except Exception:
                print("y contains unsupported datatypes for target!")
                print(y.head(10))
                raise ValueError("Target column could not be cast to float or days. Fix your source target.")
            return y

        y_train = clean_target(y_train)
        y_test  = clean_target(y_test)


            # Only select columns of dtype object or that truly look like dates
        original_date_cols = [col for col in X_train.columns
                            if any(word in col.lower() for word in ["date", "dob", "doj", "joined", "created", "timestamp"])
                            and X_train[col].dtype == 'object'] 
        # Or, more robust: and not col.endswith(("_year", "_month", "_day", "_dow", "_week"))

        for col in original_date_cols:
            for df in [X_train, X_test]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    df[f"{col}_year"] = df[col].dt.year
                    df[f"{col}_month"] = df[col].dt.month
                    df[f"{col}_day"] = df[col].dt.day
                    df[f"{col}_dow"] = df[col].dt.dayofweek
                    # Optionally: df[f"{col}_week"] = df[col].dt.isocalendar().week
                    df.drop(columns=[col], inplace=True)

        print("\n🌲 Training Random Forest model...")
        rf_model, rf_train_metrics, rf_test_metrics = train_rf_model(X_train, y_train, X_test, y_test)
        with open("mansi/target_config.json", "r") as f:
            target_col = json.load(f)["target_column"]

        # ===== FIX FOR NaN VALUES BEFORE MODEL TRAINING =====
        print("\n🔥 Training Gradient Boosting model...")
        # Drop rows in X_train/X_test that have NaNs and sync y accordingly
        X_train = X_train.dropna()
        X_test = X_test.dropna()
        y_train = y_train.loc[X_train.index]
        y_test = y_test.loc[X_test.index] 

        
        gbm_model, gbm_used_type, gbm_test_metrics = train_gbm_model(X_train, y_train, X_test, y_test, target_col)
        print("\n🤖 Comparing RF vs GBM...")

        rf_r2 = rf_test_metrics["r2"]
        gbm_r2 = gbm_test_metrics["R2"]

        if gbm_r2 > rf_r2:
            print(f"\n✅ GBM outperformed RF (GBM R²: {gbm_r2:.4f} > RF R²: {rf_r2:.4f})")
            final_model = gbm_model
            used_model_type = gbm_used_type
            selected_train_r2 = r2_score(y_train, gbm_model.predict(X_train))  # Needed for fallback check
            selected_test_r2 = gbm_r2
        else:
            print(f"\n✅ RF performed better or equally well (RF R²: {rf_r2:.4f} ≥ GBM R²: {gbm_r2:.4f})")
            final_model = rf_model
            used_model_type = "random_forest"
            selected_train_r2 = rf_train_metrics["r2"]
            selected_test_r2 = rf_test_metrics["r2"]

        #==   =============================================================================================================
        #===============================================================================================================
        #===============================================================================================================
        #===============================================================================================================
        #===============================================================================================================
        # 🔁 Optional FLAML Fallback Trainer Function
        def train_flaml_fallback(X_train, y_train, X_test, y_test, rf_test_metrics):
            warnings.filterwarnings("ignore")
            print("\n🚀 Training fallback model using FLAML...")

            automl = AutoML()
            num_rows = X_train.shape[0]
            target_std = y_train.std()
            rf_r2_score = rf_test_metrics["r2"]

            fallback_time_budget = 60

            if num_rows < 500:
                fallback_time_budget += 20
            elif num_rows < 2000:
                fallback_time_budget += 40
            else:
                fallback_time_budget += 60

            if target_std > 10:
                fallback_time_budget += 30
            elif target_std > 5:
                fallback_time_budget += 20
            else:
                fallback_time_budget += 10

            if rf_r2_score < 0.3:
                fallback_time_budget += 60
            elif 0.3<rf_r2_score < 0.5:
                fallback_time_budget += 30

            fallback_time_budget = min(fallback_time_budget, 300)
            print(f"⏳ Final FLAML Time Budget Set To: {fallback_time_budget} seconds")

            automl_settings = {
                "time_budget": fallback_time_budget,
                "metric": 'r2',
                "task": 'regression',
                "log_file_name": "flaml_fallback.log",
                "estimator_list": ["lgbm", "xgboost"],
                "seed": 42,
                "early_stop": True,
                "max_iter": 300  # Optional cap to avoid infinite loops

            }

            start_time = time.time()
            automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
            end_time = time.time()

            training_time = round(end_time - start_time, 2)
            minutes = int(training_time // 60)
            seconds = round(training_time % 60, 1)
            print(f"\n⏱️ FLAML Training Time: {minutes} min {seconds} sec")

            train_preds = automl.predict(X_train)
            test_preds = automl.predict(X_test)

            print("\n📌 FLAML used model:", type(automl.model.estimator).__name__)
            print("\n🧪 Train Target Stats:")
            print(y_train.describe())

            print("\n🧪 Test Target Stats:")
            print(y_test.describe())

            print("\n🧪 Feature Mean Differences (Train - Test):")
            print((X_train.mean() - X_test.mean()).sort_values(ascending=False))

            train_metrics = {
                "r2": round(r2_score(y_train, train_preds), 4),
                "mae": round(mean_absolute_error(y_train, train_preds), 4),
                "rmse": round(np.sqrt(mean_squared_error(y_train, train_preds)), 4)
            }

            test_metrics = {
                "r2": round(r2_score(y_test, test_preds), 4),
                "mae": round(mean_absolute_error(y_test, test_preds), 4),
                "rmse": round(np.sqrt(mean_squared_error(y_test, test_preds)), 4)
            }


            joblib.dump(automl, "fallback_flaml_model.pkl")

            print("\n📉 Fallback Model Performance:")
            print(f"   - Test R²:  {test_metrics['r2']}")
            print(f"   - MAE:      {test_metrics['mae']}")
            print(f"   - RMSE:     {test_metrics['rmse']}")

            with open("fallback_flaml_metrics.json", "w") as f:
                json.dump({
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics,
                    "training_time": training_time,
                    "fallback_time_budget": fallback_time_budget,
                    "best_estimator": str(automl.best_estimator)
                }, f, indent=4)

            return automl, train_metrics, test_metrics


        # 🔥 Use the fallback if selected model underperforms or overfits
        print("\n🔎 Checking if fallback model is needed...")

        gap = round(selected_train_r2 - selected_test_r2, 4)

        if selected_test_r2 < 0.5 or (gap > 0.15 and selected_test_r2 < 0.5):
            print(f"\n⚠️ Selected model underperformed or overfitting detected (gap = {gap})")
            fallback_model, fallback_train_metrics, fallback_test_metrics = train_flaml_fallback(
                X_train, y_train, X_test, y_test, {"r2": selected_test_r2}
            )
            final_model = fallback_model
            used_model_type = "flaml_fallback"
            print("\n✅ Fallback model trained and selected.")
        else:
            print(f"\n✅ Selected model ({used_model_type}) performed well. No fallback needed.")

        # ================= Final Saving & Summary =======================
        print("\n💾 Saving final model and summary...")

        # Save final model
        joblib.dump(final_model, "final_model.pkl")

        # Load SHAP top 5 if it exists
        shap_top5 = {}
        if os.path.exists("rf_shap_importance.json"):
            with open("rf_shap_importance.json") as f:
                shap_top5 = json.load(f)
        elif os.path.exists("gbm_shap_importance.json"):
            with open("gbm_shap_importance.json") as f:
                shap_top5 = json.load(f)

        # Decide final metrics to store
        if used_model_type == "random_forest":
            selected_train_metrics = rf_train_metrics
            selected_test_metrics = rf_test_metrics
        elif used_model_type == "gbm":
            selected_train_metrics = {
                "r2": round(r2_score(y_train, gbm_model.predict(X_train)), 4)
            }
            selected_test_metrics = gbm_test_metrics
        else:
            selected_train_metrics = fallback_train_metrics
            selected_test_metrics = fallback_test_metrics

        # Save final summary
        final_summary = {
            "model_used": used_model_type,
            "train_metrics": selected_train_metrics,
            "test_metrics": selected_test_metrics,
            "shap_top_5": shap_top5
        }

        with open("final_model_summary.json", "w") as f:
            json.dump(final_summary, f, indent=4)

        print("\n📁 Final model saved to: final_model.pkl")
        print("📄 Final summary saved to: final_model_summary.json")

        def run_full_analysis(model, X_train, y_train, X_test, y_test, model_name):

        # ------------------ Segment-wise Analysis ------------------
            print("\n📊 Performing segment-wise analysis on training data...")

            # Combine features, target, and predictions into one DataFrame
            df_full = pd.concat([X_train.copy(), y_train.copy()], axis=1)
            df_full.columns = list(X_train.columns) + ['target']
            df_full[f"{model_name}_pred"] = model.predict(X_train)
            test_preds = model.predict(X_test)

            # Define potential segment columns (you can customize this)
            segment_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            # Optional: exclude unwanted columns
            ignore_cols = {'Unnamed: 0', 'id', 'ID'}
            segment_columns = [col for col in segment_columns if col not in ignore_cols]


            # Create folder for plots if it doesn't exist
            os.makedirs(f"plots/segments_{model_name}", exist_ok=True)

            for col in segment_columns:
                if col not in df_full.columns:
                    continue

                results = []
                for val, group in df_full.groupby(col):
                    if len(group) < 3:
                        continue
                    results.append({
                        col: val,
                        "Row_Count": len(group),
                        "MAE": mean_absolute_error(group["target"], group[f"{model_name}_pred"]),
                        "MSE": mean_squared_error(group["target"], group[f"{model_name}_pred"]),
                        "R2_Score": r2_score(group["target"], group[f"{model_name}_pred"])
                    })

                result_df = pd.DataFrame(results)
                print(f"\n📊 Segment-wise analysis for: '{col}'")
                print(tabulate(result_df, headers='keys', tablefmt='fancy_grid'))

                # Save MAE plot per segment
                plt.figure(figsize=(8, 4))
                sns.barplot(data=result_df, x=col, y="MAE", palette="Blues_d", edgecolor="black")
                plt.title(f"Segment-wise MAE for {col} ({model_name})")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(f"plots/segments_{model_name}/segment_{col}_mae.png")
                plt.close()
                # =================== AFTER SEGMENT LOOP COMPLETES ===================

                print("\n📊 Creating Test Scatter & Residual Plots...")
                try:
                    test_preds = model.predict(X_test)  # ✅ Predict outside the segment loop
                    plt.figure(figsize=(6, 4))
                    sns.scatterplot(x=y_test, y=test_preds, alpha=0.5)
                    plt.xlabel("Actual")
                    plt.ylabel("Predicted")
                    plt.title("Test: Actual vs Predicted")
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(f"plots/test_actual_vs_pred_{model_name}.png")
                    plt.close()

                    test_residuals = y_test - test_preds
                    plt.figure(figsize=(6, 4))
                    sns.scatterplot(x=test_preds, y=test_residuals, alpha=0.5)
                    plt.axhline(0, color='red', linestyle='--')
                    plt.xlabel("Predicted")
                    plt.ylabel("Residuals")
                    plt.title("Test: Residual Plot")
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(f"plots/test_residual_{model_name}.png")
                    plt.close()
                    print("✅ Residual and Actual-vs-Predicted plots saved.")
                except Exception as e:
                    print(f"⚠️ Failed to generate scatter/residual plots: {e}")

        #=========================end of segment analysis====================================================
                # ------------------ Feature Importance ------------------
            try:
                    print("\n📊 Saving feature importance plot...")
                    importances = model.feature_importances_
                    # Use model's feature_names_in_ if available (scikit-learn 1.0+), else fall back to X_train.columns
                    if hasattr(model, "feature_names_in_"):
                        features = model.feature_names_in_
                    else:
                        features = X_train.columns

                    importances = model.feature_importances_

                    # Safety: ensure equal length, else print for debug and raise clear error
                    if len(features) != len(importances):
                        print("Feature names:", features)
                        print("Importances:", importances)
                        raise ValueError(f"Length mismatch: {len(features)} features, {len(importances)} importances!")

                    importance_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)


                    plt.figure(figsize=(10, 5))
                    sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis")
                    plt.title("Feature Importance")
                    plt.tight_layout()
                    os.makedirs("plots", exist_ok=True)
                    plt.savefig(f"plots/feature_importance_{model_name}.png")
                    plt.close()
                    print(f"💾 Feature importance plot saved to plots/feature_importance_{model_name}.png")
            except AttributeError:
                    print("⚠️ Skipping feature importance — model does not support feature_importances_")

            
            print("🔍 Starting SHAP...")

            
            try:
                os.environ["JOBLIB_MULTIPROCESSING"] = "0"
                print("\n⚙️ Running fast SHAP analysis with TreeExplainer...")

                X_sampled = X_train.sample(n=100, random_state=42) if len(X_train) > 100 else X_train.copy()

                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sampled, check_additivity=False)

                # Save summary plot
                plt.figure()
                shap.summary_plot(shap_values, X_sampled, show=False)
                plt.tight_layout()
                os.makedirs("plots", exist_ok=True)
                plt.savefig(f"plots/shap_summary_{model_name}.png")
                plt.close()
                print(f"\n💾 SHAP summary plot saved to plots/shap_summary_{model_name}.png")

                # Ask user for how they want SHAP displayed
                print("\n📌 How do you want SHAP insights displayed?")
                print("   1. Text Summary")
                print("   2. Tabular Summary")
                print("   3. Both")
                choice = input("Enter your choice (1/2/3): ").strip()

                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                shap_df = pd.DataFrame({
                    "Feature": X_sampled.columns,
                    "Mean |SHAP|": mean_abs_shap
                }).sort_values(by="Mean |SHAP|", ascending=False).reset_index(drop=True)

                if choice in ["1", "3"]:
                    print("\n🧠 Top 5 Important Features (Text Summary):")
                    for i, row in shap_df.head(5).iterrows():
                        print(f"   {i+1}. {row['Feature']} — avg impact: {round(row['Mean |SHAP|'], 4)}")
                    print(f"\n🗣️ Key Insight: '{shap_df.iloc[0]['Feature']}' had the most influence.")

                if choice in ["2", "3"]:
                    print("\n📊 Top 5 Features (Tabular):")
                    print(tabulate(shap_df.head(5), headers='keys', tablefmt='fancy_grid'))

                # Save to JSON
                top_5_shap = shap_df.head(5).set_index("Feature")["Mean |SHAP|"].to_dict()
                with open(f"{model_name}_shap_importance.json", "w") as f:
                    json.dump(top_5_shap, f, indent=4)
                print(f"\n💾 Saved top 5 SHAP features to {model_name}_shap_importance.json")

            except Exception as e:
                print("\n⚠️ SHAP analysis failed gracefully. Continuing without SHAP.")
                print(f"   → Reason: {str(e)}")

        # ========================🔍 Conflict Detection: Stats vs SHAP======================================================
            print("\n🧠 Comparing statistical patterns vs SHAP importance...")

            conflicts = []

            # === 1. Load SHAP scores ===
            if os.path.exists(f"{model_name}_shap_importance.json"):
                with open(f"{model_name}_shap_importance.json") as f:
                    shap_scores = json.load(f)
            else:
                shap_scores = {}
                print("⚠️ No SHAP scores found.")

            # === 2. Compute Pearson correlation with target ===
            if 'target' in df_full.columns:
                corr_matrix = df_full.corr(numeric_only=True)
                pearson_corrs = corr_matrix["target"].drop("target", errors='ignore').abs().to_dict()
            else:
                pearson_corrs = {}

            # === 3. Run T-tests for categorical cols vs target ===
            categorical_cols = df_full.select_dtypes(include=['category', 'object']).columns.tolist()
            ttest_results = {}

            for col in categorical_cols:
                if col in df_full.columns:
                    groups = [group["target"].dropna() for _, group in df_full.groupby(col)]
                    try:
                                if len(groups) == 2:
                                    stat, p = ttest_ind(groups[0], groups[1], equal_var=False)
                                elif len(groups) > 2:
                                    stat, p = f_oneway(*groups)
                                else:
                                    continue
                                ttest_results[col] = round(p, 4)
                    except:
                                continue


            # === 4. Compare & detect conflicts ===
            for feat, shap_val in shap_scores.items():
                shap_val = round(shap_val, 4)
                if shap_val < 0.01:
                    if feat in pearson_corrs and pearson_corrs[feat] > 0.3:
                        conflicts.append({
                            "feature": feat,
                            "shap": shap_val,
                            "pearson_corr": round(pearson_corrs[feat], 4),
                            "conflict": "High correlation, low SHAP"
                        })
                    elif feat in ttest_results and ttest_results[feat] < 0.05:
                        conflicts.append({
                            "feature": feat,
                            "shap": shap_val,
                            "ttest_pval": ttest_results[feat],
                            "conflict": "Significant t-test, low SHAP"
                        })

            # === 5. Save result ===
            if conflicts:
                with open("rf_conflicts.json", "w") as f:
                    json.dump(conflicts, f, indent=4)
                print("⚠️ Conflict(s) saved to rf_conflicts.json")

                # === Optional Visual Diagnostics for Conflict Features ===
                print("\n🖼️ Visualizing conflicting features and checking collinearity...")
                os.makedirs("plots/conflict_analysis", exist_ok=True)

                for c in conflicts:
                    feature = c["feature"]
                    if feature not in df_full.columns:
                        continue

                    # === 1. Plot: Scatter or Boxplot ===
                    try:
                        if pd.api.types.is_numeric_dtype(df_full[feature]):
                            plt.figure(figsize=(6,4))
                            sns.scatterplot(x=df_full[feature], y=df_full["target"])
                            plt.title(f"{feature} vs Target (Conflict Scatter)")
                            plt.xlabel(feature)
                            plt.ylabel("Target")
                            plt.tight_layout()
                            plt.savefig(f"plots/conflict_analysis/{feature}_scatter.png")
                            plt.close()
                        else:
                            plt.figure(figsize=(6,4))
                            sns.boxplot(x=df_full[feature], y=df_full["target"])
                            plt.title(f"{feature} vs Target (Conflict Boxplot)")
                            plt.xlabel(feature)
                            plt.ylabel("Target")
                            plt.tight_layout()
                            plt.savefig(f"plots/conflict_analysis/{feature}_boxplot.png")
                            plt.close()
                    except Exception as e:
                        print(f"⚠️ Failed to plot for {feature}: {e}")

                    # === 2. Correlation with other features ===
                    if pd.api.types.is_numeric_dtype(df_full[feature]):
                        try:
                            corrs = df_full.corr(numeric_only=True)[feature].drop(feature, errors='ignore').sort_values(ascending=False)
                            top_corrs = corrs[abs(corrs) > 0.5].head(3)
                            if not top_corrs.empty:
                                print(f"\n📎 {feature} is strongly correlated with:")
                                for f, v in top_corrs.items():
                                    print(f"   → {f} (corr = {round(v, 3)})")
                        except Exception as e:
                            print(f"⚠️ Failed to compute correlations for {feature}: {e}")

            else:
                print("✅ No conflicts found between stats and SHAP.")

                
            
        run_full_analysis(final_model, X_train, y_train, X_test, y_test, model_name=used_model_type)

        # =========================
        # 🧠 INTERACTIVE PREDICTION MODE
        # =========================
        if __name__ == "__main__":
            print("\n🧪 Do you want to make predictions using your trained model?")
            choice = input("Type 'yes' to continue, or anything else to skip: ").strip().lower()

            if choice == "yes":
                print("\n📌 Choose input mode:")
                print("1. Upload CSV file for batch prediction")
                print("2. Manually enter values for a single prediction")
                mode = input("Enter 1 or 2: ").strip()

                # Load column types and final model
                with open("../mansi/final_column_types.json", "r") as f:
                    column_types = json.load(f)

                with open("mansi/target_config.json", "r") as f:
                    target_col = json.load(f)["target_column"]

                model = joblib.load("final_model.pkl")

                if mode == "1":
                    file_path = input("\n📂 Enter the full path to your CSV file: ").strip()

                    if not os.path.exists(file_path):
                        print(f"❌ File not found: {file_path}")
                    else:
                        df_input = pd.read_csv(file_path)
                        df_input = df_input.drop(columns=[target_col], errors="ignore")
                        preds = model.predict(df_input)
                        df_input["Predicted"] = preds
                        df_input.to_csv("predicted_rf.csv", index=False)
                        print("\n✅ Predictions saved to: predicted_rf.csv")

                elif mode == "2":
                    print("\n🧾 Manual Input Mode:")

                    input_data = {}
                    encoders_path = "mansi/encoders"
                    encoders = {}

                    # Preload all encoders
                    for col, dtype in column_types.items():
                        if col == target_col:
                            continue
                        if dtype == "category":
                            encoder_file = os.path.join(encoders_path, f"{col}_encoder.pkl")
                            if os.path.exists(encoder_file):
                                encoders[col] = joblib.load(encoder_file)
                            else:
                                encoders[col] = None  # Still store as None for fallback

                    # Now loop through columns and ask input
                    for col, dtype in column_types.items():
                        if col == target_col:
                            continue

                        print(f"\n👉 {col} (type: {dtype})")

                        if dtype == "category":
                            le = encoders.get(col)
                            if le:
                                valid_classes = list(le.classes_)
                                print(f"   🔹 Valid options: {valid_classes}")
                            else:
                                valid_classes = []
                                print("   ⚠️ No encoder found — skipping category validation.")

                        while True:
                            value = input("   Your input: ").strip()
                            try:
                                if dtype == "numeric":
                                    input_data[col] = float(value)
                                elif dtype == "ordinal":
                                    input_data[col] = int(value)
                                elif dtype == "category":
                                    le = encoders.get(col)
                                    if le is None:
                                        input_data[col] = value
                                        break
                                    if value.lower() not in le.classes_:
                                        print(f"❌ Invalid input. Must be one of: {list(le.classes_)}")
                                        continue
                                    input_data[col] = le.transform([value.lower()])[0]
                                elif dtype == "time":
                                    input_data[col] = pd.to_datetime(value)
                                elif dtype == "id":
                                    input_data[col] = str(value)
                                else:
                                    input_data[col] = value
                                break
                            except Exception as e:
                                print(f"❌ Error: {e}. Try again.")

                    df_manual = pd.DataFrame([input_data])
                    pred = model.predict(df_manual)[0]
                    print(f"\n🎯 Predicted Value: {round(pred, 2)}")
