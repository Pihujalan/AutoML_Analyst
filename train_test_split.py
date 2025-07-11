import pandas as pd
from sklearn.model_selection import train_test_split
df=pd.read_csv("cleaned_input.csv")
print("Available columns in the dataset: ")
print(df.columns.tolist())
target_col = input("\n Enter the name of the target column from the above list: ")
if target_col not in df.columns:
    print(f" Column '{target_col}' not found in the dataset. Exiting.")
    exit()

X = df.drop(columns=[target_col])
Y = df[target_col]
print(f" Features and target separated. Shape of X: {X.shape}, Y: {Y.shape}")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
Y_train.to_csv("Y_train.csv", index=False)
Y_test.to_csv("Y_test.csv", index=False)

print("\n✅ Data split completed:")
print(f"   - X_train: {X_train.shape}")
print(f"   - X_test: {X_test.shape}")
print(f"   - Y_train: {Y_train.shape}")
print(f"   - Y_test: {Y_test.shape}")