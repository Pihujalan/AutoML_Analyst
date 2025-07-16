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
import pprint
from shap import TreeExplainer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
from scipy.stats import skew, kurtosis, zscore
import argparse
import sys
#====================================*Pattern Analysis*==============================================================
def run_full_pipeline(cleaned_csv_path="data/cleaned_input.csv", target_col=None):
    import pandas as pd
    import os
    import json
    if not os.path.exists("plots"):
        os.mkdir("plots")

    with open("../mansi/final_column_types.json", "r") as f:
        final_column_types = json.load(f)
    # Load dataset
    #df = pd.read_csv("mansi/cleaned_input.csv")
    df = pd.read_csv(cleaned_csv_path)

    # --- Missing Value Pattern Analysis ---
    print("📊 Running Missing Value Pattern Analysis...")

    missing_summary = pd.DataFrame({
        "column_name": df.columns,
        "num_missing": df.isnull().sum().values,
        "percent_missing": df.isnull().mean().values * 100
    })

        # ✅ Ensure target output folder exists
    # ✅ Use current working directory (safe for Streamlit)
    cwd = os.getcwd()
    output_dir = os.path.join(cwd, "mansi")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "missing_summary.csv")
    missing_summary.to_csv(output_path, index=False)
    print(f"✅ Missing value summary saved to {output_path}")
    for col, col_type in final_column_types.items():
        if col_type.lower() == "numeric":
            if col in df.columns:
            #HISTOGRAM
                plt.figure(figsize=(8,5))
                sns.histplot(df[col])
                plt.title(f"Histogram for {col}")
                plt.xlabel(col)
                plt.ylabel("Frequency")
                plt.savefig(f"plots/{col}_histogram.png")
                plt.close()

            #BOXPLOT
                plt.figure(figsize=(8,5))
                sns.boxplot(x=df[col])
                plt.title(f"Boxplot for {col}")
                plt.xlabel(col)
                plt.savefig(f"plots/{col}_boxplot.png")
                plt.close()

    numeric_columns = [col for col, col_type in final_column_types.items() if col_type.lower() == "numeric"]
    category_columns=[col for col,col_type in final_column_types.items() if col_type.lower()=="category"]

    for num_col in numeric_columns:
        for cat_col in category_columns:
            if num_col in df.columns and cat_col in df.columns:
                plt.figure(figsize=(8,5))
                sns.boxplot(x=df[cat_col], y=df[num_col])
                plt.title(f"{num_col} by {cat_col}")
                plt.xlabel(cat_col)
                plt.ylabel(num_col)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f"plots/{num_col}_by_{cat_col}_boxplot.png")
                plt.close()

    for cat_col in category_columns:
        if cat_col in df.columns:
            
            counts = df[cat_col].value_counts().reset_index()
            counts.columns = [cat_col, 'count']
            counts = counts.sort_values(by='count', ascending=False)

            plt.figure(figsize=(8,5))
            ax = sns.barplot(x=cat_col,y='count',data=counts)
            for p in ax.patches:
                    ax.annotate(format(p.get_height(), '.0f'),
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center',
                                xytext=(0, 5),
                                textcoords='offset points')

            plt.title(f"Barplot for {cat_col}")
            plt.xlabel(cat_col)
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"plots/barplot_{cat_col}.png")
            plt.close()

    for col, col_type in final_column_types.items():
        if col_type.lower() == "category":
            if col in df.columns:
                plt.figure(figsize=(8,5))
                sns.countplot(x=df[col])
                plt.title(f"Countplot for {col}") 
                plt.xlabel(col)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f"plots/countplot_{col}.png")
                plt.close()

    for cat_col in category_columns:
        for num_col in numeric_columns:
            if num_col in df.columns and cat_col in df.columns:
                plt.figure(figsize = (8,5))
                sns.violinplot(x=df[cat_col],y=df[num_col])
                plt.title(f"Violinplot for {num_col} vs {cat_col}")
                plt.xlabel(cat_col)
                plt.ylabel(num_col)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f"plots/violinplot_{num_col}_by_{cat_col}.png")
                plt.close()

    # SCATTER PLOTS
    for i in range(len(numeric_columns)):
        for j in range(i + 1, len(numeric_columns)):
            col_x = numeric_columns[i]
            col_y = numeric_columns[j]
            
            if col_x in df.columns and col_y in df.columns:
                plt.figure(figsize=(8, 5))
                sns.scatterplot(x=df[col_x], y=df[col_y])
                plt.title(f"Scatterplot: {col_x} vs {col_y}")
                plt.xlabel(col_x)
                plt.ylabel(col_y)
                plt.tight_layout()
                safe_x = col_x.replace(" ", "_").replace("/", "_")
                safe_y = col_y.replace(" ", "_").replace("/", "_")
                plt.savefig(f"plots/scatterplot_{safe_x}_vs_{safe_y}.png")
                plt.close()


    # HEATMAP OF CORRELATIONS
    if len(numeric_columns) > 1:
        plt.figure(figsize=(10, 8))
        corr_matrix = df[numeric_columns].corr()

        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, linewidths=0.5)
        plt.title("Correlation Heatmap of Numeric Columns")
        plt.tight_layout()
        plt.savefig("plots/correlation_heatmap.png")
        plt.close()
    time_columns = [col for col, col_type in final_column_types.items() if col_type.lower() in ["time", "datetime"]]

    for time_col in time_columns:
        for num_col in numeric_columns:
            if time_col in df.columns and num_col in df.columns:
                plt.figure(figsize=(10, 5))
                df_sorted = df.sort_values(by=time_col)
                sns.lineplot(x=df_sorted[time_col], y=df_sorted[num_col])
                plt.title(f"Time Series: {num_col} over {time_col}")
                plt.xlabel(time_col)
                plt.ylabel(num_col)
                plt.tight_layout()
                plt.savefig(f"plots/timeseries_{num_col}_by_{time_col}.png")
                plt.close()
                
    print("\n📉 Outlier Detection (IQR Method):")
    outlier_rows = set()

    for col in numeric_columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            num_outliers = outliers.shape[0]
            print(f"Column: {col} | Outliers: {num_outliers}")

            outlier_rows.update(outliers.index.tolist())

    # Save all rows with at least one outlier
    if outlier_rows:
        outlier_df = df.loc[list(outlier_rows)]
        outlier_df.to_csv("outlier_rows.csv", index=False)
        print(f"✅ Saved {len(outlier_df)} outlier rows to 'outlier_rows.csv'")
    else:
        print("✅ No outliers detected.")

    print("\nCardinality analysis for categorical columns:")

    encoding_suggestions = {}

    for col in category_columns:
        if col in df.columns:
            unique_count = df[col].nunique()
            print(f"Column: {col} | Unique values: {unique_count}")
            if unique_count <= 10:
                suggestion = "One-Hot Encoding"
            elif unique_count <= 50:
                suggestion = "Frequency Encoding"
            else:
                suggestion = "Target Encoding"
            
            encoding_suggestions[col] = {
                "unique_values": unique_count,
                "recommended_encoding": suggestion
            }
    print(json.dumps(encoding_suggestions, indent=4))

    # === Pihu's Additional Functions ===

    def classify_columns(df):
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
        return numeric, categorical

    # Pearson correlation (linear numeric)

    def run_statistical_tests(df, column_types, target_col=None):
        print("\n Statistical Testing Summary")
        print("="*35)

        categorical_cols = [col for col, t in column_types.items() if t == 'category']
        numeric_cols = [col for col, t in column_types.items() if t in ['numeric', 'ordinal']]

        for cat_col in categorical_cols:
            # Chi-squared Test for Categorical vs Categorical
            for other_cat in categorical_cols:
                if cat_col != other_cat:
                    ctab = pd.crosstab(df[cat_col], df[other_cat])
                    if ctab.shape[0] > 1 and ctab.shape[1] > 1:
                        chi2, p, _, _ = chi2_contingency(ctab)
                        result = " Significant" if p < 0.05 else " Not significant"
                        print(f"\n Chi-Square: {cat_col} vs {other_cat}")
                        print(f"• p-value: {p:.4f} → {result}")

            # T-test or ANOVA for Categorical vs Numeric
            for num_col in numeric_cols:
                groups = [group[num_col].dropna() for name, group in df.groupby(cat_col)]
                if len(groups) == 2:
                    stat, p = ttest_ind(groups[0], groups[1], equal_var=False)
                    result = " Significant" if p < 0.05 else " Not significant"
                    print(f"\n T-Test: {num_col} by {cat_col}")
                    print(f"• p-value: {p:.4f} → {result}")
                elif len(groups) > 2:
                    stat, p = f_oneway(*groups)
                    result = " Significant" if p < 0.05 else " Not significant"
                    print(f"\n ANOVA: {num_col} by {cat_col}")
                    print(f"• p-value: {p:.4f} → {result}")

    #print output:
    column_types = {
        col: 'numeric' if df[col].dtype in ['int64', 'float64'] else 'category'
        for col in df.columns
    }

    run_statistical_tests(df, column_types)

    def analyze_time_patterns(df, time_col_name=None, target_cols=[]):
        """
        Generalized time-based pattern analysis:
        - Detects datetime column (or accepts user-specified one)
        - Extracts year from datetime
        - Analyzes trend of target columns over years
        - Handles leave/stay logic if binary target is present
        """

        #  Try to auto-detect the datetime column if not provided
        if not time_col_name:
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower() or 'year' in col.lower():
                    time_col_name = col
                    break
            if not time_col_name:
                print("No valid time-based column found.")
                return

        #  Convert to datetime and extract year
        df[time_col_name] = pd.to_datetime(df[time_col_name], errors='coerce')
        df['year'] = df[time_col_name].dt.year

        #  Check if there's enough time variation to proceed
        if df['year'].nunique() < 5:
            print(" Not enough unique years for meaningful time-based analysis.")
            return

        print(f"\n Time-Based Analysis Using '{time_col_name}' Column\n{'-'*60}")

        #  Default numeric targets if not specified
        if not target_cols:
            numeric_cols = df.select_dtypes(include=['int64', 'float64', 'Int64']).columns.tolist()
            target_cols = [col for col in numeric_cols if col != 'year']

        #  Generate year-wise summaries for numeric columns
        if target_cols:
            year_summary = df.groupby('year')[target_cols].agg(['mean', 'count']).round(2)
            print(" Year-wise Summary (Numeric Averages & Counts):")
            print(year_summary)
        else:
            print(" No numeric targets found for summary.")

        #  Auto-detect binary classification column (like 'leaveornot')
        binary_col = None
        for col in df.select_dtypes(include='object').columns:
            if df[col].nunique() == 2:
                binary_col = col
                break

        #  If a binary target like 'leaveornot' exists, show trend
        if binary_col:
            print(f"\n '{binary_col}' Distribution by Year:")
            leave_counts = df.groupby(['year', binary_col]).size().unstack(fill_value=0)
            print(leave_counts)
    # print output:
    analyze_time_patterns(df)

    #Outlier Detection

    # Descriptive Statistics Summary
    def descriptive_stats(df, column_types):
        summary = []

        for col in df.columns:
            col_data = df[col]
            col_type = column_types.get(col, None)
            result = {
                'Column': col,
                'Type': col_type if col_type else str(col_data.dtype),
                'Count': col_data.count(),
                'Missing': df[col].isna().sum(),
                'Unique': col_data.nunique()
            }

            if col_type == "numeric":
                result.update({
                    'Mean': col_data.mean(),
                    'Median': col_data.median(),
                    'Mode': col_data.mode().iloc[0] if not col_data.mode().empty else None,
                    'Std': col_data.std(),
                    'Min': col_data.min(),
                    '25%': col_data.quantile(0.25),
                    '50%': col_data.quantile(0.5),
                    '75%': col_data.quantile(0.75),
                    'Max': col_data.max(),
                    'Skewness': skew(pd.to_numeric(col_data, errors='coerce').dropna()),
                    'Kurtosis': kurtosis(pd.to_numeric(col_data, errors='coerce').dropna()),
                })

            elif col_type == "ordinal":
                result.update({
                    'Mode': col_data.mode().iloc[0] if not col_data.mode().empty else None,
                    'Min': col_data.min(),
                    'Max': col_data.max()
                })

            elif col_type == "category":
                mode = col_data.mode().iloc[0] if not col_data.mode().empty else None
                freq = col_data.value_counts().iloc[0] if not col_data.value_counts().empty else None
                result.update({
                    'Top': mode,
                    'Freq': freq
                })

            elif col_type == "time":
                result.update({
                    'Min Date': col_data.min(),
                    'Max Date': col_data.max()
                })

            elif col_type == "id":
                # We can skip detailed stats for ID-like columns
                pass

            else:
                # fallback for unclassified or unknown
                if pd.api.types.is_numeric_dtype(col_data):
                    result.update({
                        'Mean': col_data.mean(),
                        'Std': col_data.std()
                    })
                elif pd.api.types.is_datetime64_any_dtype(col_data):
                    result.update({
                        'Min Date': col_data.min(),
                        'Max Date': col_data.max()
                    })

            summary.append(result)

        return pd.DataFrame(summary)
    # make the statistical output readable:
    def print_descriptive_summary_pretty(summary_df):
        type_groups = summary_df.groupby('Type')

        for col_type, group in type_groups:
            print(f"\n{col_type.upper()} COLUMNS:\n" + "-" * 40)
            for _, row in group.iterrows():
                print(f"\n {row['Column']}")
                print(f" Count: {row['Count']}")
                print(f" Unique: {row['Unique']}")
                print(f" Missing: {row['Missing']}")

                if col_type == "numeric":
                    print(f" Mean: {row['Mean']:.2f}")
                    print(f" Median: {row['Median']}")
                    print(f" Mode: {row['Mode']}")
                    # Std Dev - handle if it's numeric only
                    std_dev = row.get('Std_Dev')
                    if std_dev is not None:
                        if isinstance(std_dev, (int, float, np.number)):
                            print(f" Std Dev: {std_dev:.2f}")
                        else:
                            print(f" Std Dev: {std_dev}")
                    print(f" Min: {row['Min']}")
                    print(f" 25%: {row['25%']}")
                    print(f" 50%: {row['50%']}")
                    print(f" 75%: {row['75%']}")
                    print(f" Max: {row['Max']}")
                    print(f" Skewness: {row['Skewness']:.2f}")
                    print(f" Kurtosis: {row['Kurtosis']:.2f}")

                elif col_type == "ordinal":
                    print(f" Mode: {row['Mode']}")
                    print(f" Min: {row['Min']}")
                    print(f" Max: {row['Max']}")

                elif col_type == "category":
                    print(f" Most Frequent: {row['Top']}")
                    print(f" Freq: {row['Freq']}")

                elif col_type == "time":
                    print(f"Min Date: {row['Min Date']}")
                    print(f"Max Date: {row['Max Date']}")

                elif col_type == "id":
                    print(f"Skipped: ID-like column")

                else:
                    print(f"No detailed analysis for this type.")

    #run the code:
    summary_df = descriptive_stats(df, column_types)
    print_descriptive_summary_pretty(summary_df)


    #Bivariate / Segment-Wise Analysis

    def compute_pearson(df, numeric_cols):
        return df[numeric_cols].corr(method='pearson')

    # Spearman correlation (ordinal/skewed)

    def auto_bivariate_all_targets(df, column_types):
        print(" Full Bivariate Insight Engine: All Columns as Target\n" + "="*60)

        for target_col in df.columns:
            print(f"\nTarget Column: {target_col} (type: {column_types.get(target_col, 'unknown')})")
            print("-"*60)

            for compare_col in df.columns:
                if compare_col == target_col:
                    continue  # Don't compare a column to itself

                compare_type = column_types.get(compare_col, 'unknown')
                target_type = column_types.get(target_col, 'unknown')

                # Case 1: Target is numeric, compare is categorical
                if target_type == 'numeric' and compare_type in ['category', 'ordinal']:
                    print(f"\n{target_col} (numeric) by {compare_col} (categorical)")
                    try:
                        grouped = df.groupby(compare_col)[target_col].agg(['mean', 'median', 'count', 'min', 'max', 'std'])
                        print(grouped)
                    except:
                        print("Could not group")

                # Case 2: Target is categorical, compare is numeric
                elif target_type in ['category', 'ordinal'] and compare_type == 'numeric':
                    print(f"\n{compare_col} (numeric) by {target_col} (categorical)")
                    try:
                        grouped = df.groupby(target_col)[compare_col].agg(['mean', 'median', 'count', 'min', 'max', 'std'])
                        print(grouped)
                    except:
                        print(" Could not group")

                # Case 3: Numeric vs numeric → correlation
                elif target_type == 'numeric' and compare_type == 'numeric':
                    try:
                        corr = df[[target_col, compare_col]].corr().iloc[0, 1]
                        print(f"\nPearson Correlation between {target_col} and {compare_col} = {round(corr, 3)}")
                    except:
                        print(" Could not compute correlation")

                # Case 4: Categorical vs categorical → cross-tab
                elif target_type in ['category', 'ordinal'] and compare_type in ['category', 'ordinal']:
                    print(f"\n Categorical Crosstab: {target_col} vs {compare_col}")
                    try:
                        cross = pd.crosstab(df[target_col], df[compare_col])
                        print(cross)
                    except:
                        print(" Could not generate cross-tab")

                # Case 5: Time series target or compare
                elif target_type == 'time' and compare_type != 'id':
                    print(f"\n Time Trend: {target_col} over {compare_col}")
                    try:
                        trend = df.groupby(df[target_col].dt.year)[compare_col].mean()
                        print(trend)
                    except:
                        print("Could not build time trend")

    #run the bivariate function:
    auto_bivariate_all_targets(df, column_types)


    #Correlation & Association Matrix

    # Automatically classify column types

    def smart_outlier_detector(df, z_thresh=3):
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        total_rows = len(df)
        report = []

        print(" Smart Outlier Detection Report")
        print(f" Analyzing columns: {list(numeric_cols)}\n")

        for col in numeric_cols:
            if df[col].nunique() < 10:
                print(f" Skipping '{col}' — too few unique values.")
                continue
            
            if df[col].isnull().all():
                print(f" Skipping '{col}' — all values are NaN.")
                continue

            col_skew = skew(df[col].dropna())
            method = "Z-Score" if abs(col_skew) < 0.5 else "IQR"

            if method == "Z-Score":
                z_scores = np.abs(zscore(df[col].fillna(df[col].mean())))
                outliers = df[z_scores > z_thresh]
            else:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = df[(df[col] < lower) | (df[col] > upper)]

            count = len(outliers)
            percent = round((count / total_rows) * 100, 2)

            report.append({
                "column": col,
                "method": method,
                "skewness": round(col_skew, 2),
                "count": count,
                "percent": percent
            })

        found = False
        for r in report:
            found = found or r['count'] > 0
            print(f" {r['column']} (Skew: {r['skewness']}) → Method: {r['method']}")
            print(f"   Outliers: {r['count']} / {total_rows} ({r['percent']}%)\n")

        if not found:
            print(" No significant outliers detected using smart method.")
    # print output:
    smart_outlier_detector(df)

    #Statistical Testing (t test, anova and chi square)

    def descriptive_stats(df, column_types):
        summary = []

        for col in df.columns:
            col_data = df[col]
            col_type = column_types.get(col, None)
            result = {
                'Column': col,
                'Type': col_type if col_type else str(col_data.dtype),
                'Count': col_data.count(),
                'Missing': df[col].isna().sum(),
                'Unique': col_data.nunique()
            }

            if col_type == "numeric":
                result.update({
                    'Mean': col_data.mean(),
                    'Median': col_data.median(),
                    'Mode': col_data.mode().iloc[0] if not col_data.mode().empty else None,
                    'Std': col_data.std(),
                    'Min': col_data.min(),
                    '25%': col_data.quantile(0.25),
                    '50%': col_data.quantile(0.5),
                    '75%': col_data.quantile(0.75),
                    'Max': col_data.max(),
                    'Skewness': skew(col_data.dropna()),
                    'Kurtosis': kurtosis(col_data.dropna())
                })

            elif col_type == "ordinal":
                result.update({
                    'Mode': col_data.mode().iloc[0] if not col_data.mode().empty else None,
                    'Min': col_data.min(),
                    'Max': col_data.max()
                })

            elif col_type == "category":
                mode = col_data.mode().iloc[0] if not col_data.mode().empty else None
                freq = col_data.value_counts().iloc[0] if not col_data.value_counts().empty else None
                result.update({
                    'Top': mode,
                    'Freq': freq
                })

            elif col_type == "time":
                result.update({
                    'Min Date': col_data.min(),
                    'Max Date': col_data.max()
                })

            elif col_type == "id":
                # We can skip detailed stats for ID-like columns
                pass

            else:
                # fallback for unclassified or unknown
                if pd.api.types.is_numeric_dtype(col_data):
                    result.update({
                        'Mean': col_data.mean(),
                        'Std': col_data.std()
                    })
                elif pd.api.types.is_datetime64_any_dtype(col_data):
                    result.update({
                        'Min Date': col_data.min(),
                        'Max Date': col_data.max()
                    })

            summary.append(result)

        return pd.DataFrame(summary)
    # make the statistical output readable:

    def compute_spearman(df, numeric_cols):
        return df[numeric_cols].corr(method='spearman')

    # Cramér's V for categorical-categorical

    def compute_cramers_v_matrix(df, categorical_cols):
        v_matrix = pd.DataFrame(index=categorical_cols, columns=categorical_cols)
        for col1 in categorical_cols:
            for col2 in categorical_cols:
                try:
                    table = pd.crosstab(df[col1], df[col2])
                    v = cramers_v(table)
                    v_matrix.loc[col1, col2] = round(v, 3)
                except:
                    v_matrix.loc[col1, col2] = np.nan
        return v_matrix

    # Assume df is your dataframe
    numeric_cols, categorical_cols = classify_columns(df)

    # Run all three correlation types
    pearson_corr = compute_pearson(df, numeric_cols)
    spearman_corr = compute_spearman(df, numeric_cols)
    cramers_corr = compute_cramers_v_matrix(df, categorical_cols)

    # Output
    print(" Pearson Correlation (Numeric):")
    print(pearson_corr.fillna("-"))

    print("\n Spearman Correlation (Ordinal/Skewed):")
    print(spearman_corr.fillna("-"))

    print("\nCramér’s V Correlation (Categorical):")
    print(cramers_corr.fillna("-"))


    #Time-Based Pattern (If time column exists)

    def classify(name,column):
        name=name.lower()
        unique= column.nunique()
        dtype=column.dtype
        if any (word in name for word in ["year", "date", "time", "dob", "doj", "joined", "created", "timestamp"]) or pd.api.types.is_datetime64_any_dtype(column):
            return "time"
        if unique==len(column):
            return "ID"
        if dtype in ['int64','float64'] and unique<=15:
            return 'ordinal'
        if dtype in ['int64','float64']:
            return 'numeric'
        if (dtype == 'object' or dtype.name== 'category') and unique<=15 :
            return 'category'
    
        return 'ID'
    print(" Column Classification Results:\n")
    for col in df.columns:
        guess = classify(col, df[col])
        dtype = df[col].dtype
        print(f"{col} ({dtype}): {guess}")

    #to save the classification  in dictionary 

    column_types = final_column_types

    # updating dtypes acc to saved classification in dictionary

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

            except Exception as e:
                print(f"Failed to convert '{col}' to '{ctype}': {e}")

        return df_cleaned

    # Run the function

    df_cleaned = force_convert_exact_dtypes(df, column_types)
    df = df_cleaned  # Make sure modeling uses cleaned version

    # Final Check
    print("Final Column Data Types:")
    print(df_cleaned.dtypes)

    # Descriptive Statistics Summary

    def cramers_v(confusion_matrix):
        chi2 = ((confusion_matrix - confusion_matrix.mean())**2 / confusion_matrix.mean()).sum().sum()
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2_corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        r_corr = r - ((r-1)**2)/(n-1)
        k_corr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2_corr / min((k_corr-1), (r_corr-1)))



    #======================================*end of pattern analysis*=====================================================
    #encode data
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # Ensure encoders folder exists
    os.makedirs("mansi/encoders", exist_ok=True)


    print("Original shape:", df.shape)

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
    # Load target column from earlier saved config
    with open("mansi/target_config.json", "r") as f:
        target_col = json.load(f)["target_column"]

    if target_col not in df.columns:
        print(f"❌ Target column '{target_col}' not found in the dataset. Exiting.")
        exit()

        

    X = df.drop(columns=[target_col])
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
    #--------------------------------GBM-----------------------------------------------------
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

        if r2 < 0.5 or (gap > 0.15 and r2 < 0.5):
            print(f"\n⚠️ GBM model underperformed or overfitting detected (gap = {gap})")

            automl = AutoML()
            automl.fit(X_train=X_train, y_train=y_train, task="regression", time_budget=60)

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

    print("\n🌲 Training Random Forest model...")
    rf_model, rf_train_metrics, rf_test_metrics = train_rf_model(X_train, y_train, X_test, y_test)
    with open("mansi/target_config.json", "r") as f:
        target_col = json.load(f)["target_column"]

    print("\n🔥 Training Gradient Boosting model...")
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
            "estimator_list": ["lgbm", "xgboost", "rf"],
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
        segment_columns = [
            'education', 'joiningyear', 'city', 'paymenttier',
            'gender', 'everbenched', 'experienceincurrentdomain', 'leaveornot'
        ]

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
                features = X_train.columns
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
            with open("mansi/final_column_types.json", "r") as f:
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
