import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd 
import os
if not os.path.exists("plots"):
    os.mkdir("plots")

with open("final_column_types.json", "r") as f:
    final_column_types = json.load(f)
df = pd.read_csv("cleaned_input.csv")
df.columns = df.columns.str.lower()

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

