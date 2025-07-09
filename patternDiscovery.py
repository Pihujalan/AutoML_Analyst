import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd 
import os
if not os.path.exists("plots"):
    os.mkdir("plots")

with open("final_column_types.json", "r") as f:
    final_column_types = json.load(f)
df = pd.read_csv("input.csv")
df.columns = df.columns.str.lower()

for col, col_type in final_column_types.items():
    if col_type.lower() == "numeric":
        if col in df.columns:
         #HISTOGRAM
         sns.histplot(df[col])
         plt.title(f"Histogram for {col}")
         plt.xlabel(col)
         plt.ylabel("Frequency")
         plt.savefig(f"plots/{col}_histogram.png")
         plt.close()

         #BOXPLOT
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
