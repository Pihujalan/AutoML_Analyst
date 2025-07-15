import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from category_encoders import TargetEncoder
import warnings
from flaml import AutoML
from tabulate import tabulate

warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set(style="whitegrid")

# Ensure plot folder exists
os.makedirs("plots", exist_ok=True)

# ------------------ Load Data ------------------
print("📥 Loading train/test datasets...")
X_train = pd.read_csv("mansi/X_train.csv")
X_test = pd.read_csv("mansi/X_test.csv")
y_train = pd.read_csv("mansi/y_train.csv").squeeze()
y_test = pd.read_csv("mansi/y_test.csv").squeeze()

with open("mansi/target_config.json", "r") as f:
    target_col = json.load(f)["target_column"]

# ------------------ Encode ------------------
print("🔡 Performing target encoding on categorical columns...")
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
te = TargetEncoder()
X_train[cat_cols] = te.fit_transform(X_train[cat_cols], y_train)
X_test[cat_cols] = te.transform(X_test[cat_cols])

joblib.dump(te, "target_encoder.pkl")

# ------------------ Baseline Model ------------------
print("🚀 Training baseline model...")
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

# ------------------ Residual & Scatter Plot ------------------
print("\n📈 Saving residual and scatter plots...")
residuals = y_test - y_pred

plt.figure(figsize=(10, 5))
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("plots/residual_distribution.png")
plt.close()

plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.title("Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.tight_layout()
plt.savefig("plots/actual_vs_predicted.png")
plt.close()

# ------------------ Feature Importance ------------------
print("\n📊 Saving feature importance plot...")
importances = best_model.feature_importances_
features = X_train.columns
importance_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("plots/feature_importance.png")
plt.close()

# ------------------ Segment-wise Analysis ------------------
print("\n📊 Performing segment-wise analysis...")
df_full = pd.concat([X_train, y_train], axis=1)
df_full[target_col] = y_train
df_full["pred"] = best_model.predict(X_train)

segment_columns = ['education', 'joiningyear', 'city', 'paymenttier', 'gender', 'everbenched', 'experienceincurrentdomain', 'leaveornot']

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
            "MAE": mean_absolute_error(group[target_col], group["pred"]),
            "MSE": mean_squared_error(group[target_col], group["pred"]),
            "R2_Score": r2_score(group[target_col], group["pred"])
        })

    result_df = pd.DataFrame(results)
    print(f"\n📊 Segment-wise analysis for: '{col}'")
    print(result_df)

    # Save MAE plot per segment
    plt.figure(figsize=(8, 4))
    sns.barplot(data=result_df, x=col, y="MAE", hue=col, palette="Blues_d", legend=False)
    plt.title(f"Segment-wise MAE for {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"plots/segment_{col}_mae.png")
    plt.close()


print("\n Checking if fallback model is needed...")

train_preds = best_model.predict(X_train)
r2_train = r2_score(y_train, train_preds)
gap = round(r2_train - r2, 4)
print(f"R² (Train): {r2_train:.2f}, R² (Test): {r2:.2f}, Gap: {gap:.4f}")

if r2 < 0.75 or (gap > 0.15 and r2 < 0.75):
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
    print(f"\FLAML Performance:\nMAE: {mae_fallback:.2f} | MSE: {mse_fallback:.2f} | R²: {r2_fallback:.2f}")
else:
    final_model = best_model
    used_model_type = "gbm"
    print("✅ GBM model performed well. No fallback needed.")

# ------------------ Save Final Model ------------------
joblib.dump(final_model, "final_model.pkl")
with open("model_type.txt", "w") as f:
    f.write(used_model_type)


def run_gbm_shap_analysis(model, X_train):
    try:
        os.environ["JOBLIB_MULTIPROCESSING"] = "0"  # Avoid SHAP loky deadlock
        print("\n⚙️ Running fast SHAP analysis with TreeExplainer...")

        X_sampled = X_train.sample(n=100, random_state=42) if len(X_train) > 100 else X_train.copy()

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sampled, check_additivity=False)
        shap.summary_plot(shap_values, X_sampled)

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

    except Exception as e:
        print("⚠️ SHAP analysis failed:", str(e))

if used_model_type == "gbm":
    run_gbm_shap_analysis(final_model, X_train)
