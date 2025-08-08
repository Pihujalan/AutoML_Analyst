import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd 
import os
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
from scipy.stats import skew, kurtosis, zscore

if not os.path.exists("plots"):
    os.mkdir("plots")

with open("data_processing/final_column_types.json", "r") as f:
    final_column_types = json.load(f)
df = pd.read_csv("data_processing/cleaned_input.csv")
df.columns = df.columns.str.lower()

# --- Missing Value Pattern Analysis ---
print("📊 Running Missing Value Pattern Analysis...")

missing_summary = pd.DataFrame({
    "column_name": df.columns,
    "num_missing": df.isnull().sum().values,
    "percent_missing": df.isnull().mean().values * 100
})

missing_summary.to_csv("data_processing/missing_summary.csv", index=False)
print("✅ Missing value summary saved to 'data_processing/missing_summary.csv'")

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

column_types = {}

for col in df.columns:
    col_type = classify(col, df[col])
    column_types[col] = col_type
df_cleaned = df.copy()
print(column_types)

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

