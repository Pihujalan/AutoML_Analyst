import pandas as pd

df = pd.read_csv("input.csv")
print("📊 Running Missing Value Pattern Analysis...")


summary = pd.DataFrame({
    "column_name": df.columns,
    "num_missing": df.isnull().sum().values,
    "percent_missing": df.isnull().mean().values * 100
})

summary.to_csv("missing_summary.csv", index=False)
print("✅ Missing value summary saved to 'missing_summary.csv'")