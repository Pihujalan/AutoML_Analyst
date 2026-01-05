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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from flaml import AutoML
import os 
from category_encoders import TargetEncoder
import pprint
from shap import TreeExplainer


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
    os.makedirs("outputs/models", exist_ok=True)
    joblib.dump(best_rf, "outputs/models/best_rf_model.pkl")

    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_train, y=train_preds, alpha=0.5)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Train: Actual vs Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    residuals = y_train - train_preds
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=train_preds, y=residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Train: Residual Plot")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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

    os.makedirs("outputs/metrics", exist_ok=True)
    with open("outputs/metrics/best_rf_metrics.json", "w") as f:
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
    print("\n📊 Performing segment-wise analysis on training data...")

    # Combine features, target, and predictions into one DataFrame
    df_full = pd.concat([X_train.copy(), y_train.copy()], axis=1)
    df_full.columns = list(X_train.columns) + ['target']
    df_full["rf_pred"] = best_rf.predict(X_train)

    # Define potential segment columns (you can customize this)
    segment_columns = [
        'education', 'joiningyear', 'city', 'paymenttier',
        'gender', 'everbenched', 'experienceincurrentdomain', 'leaveornot'
    ]

    # Create folder for plots if it doesn't exist
    os.makedirs("outputs/plots/segments_rf", exist_ok=True)

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
                "MAE": mean_absolute_error(group["target"], group["rf_pred"]),
                "MSE": mean_squared_error(group["target"], group["rf_pred"]),
                "R2_Score": r2_score(group["target"], group["rf_pred"])
            })

        result_df = pd.DataFrame(results)
        print(f"\n📊 Segment-wise analysis for: '{col}'")
        print(tabulate(result_df, headers='keys', tablefmt='fancy_grid'))

        # Save MAE plot per segment
        plt.figure(figsize=(8, 4))
        sns.barplot(data=result_df, x=col, y="MAE", palette="Blues_d", edgecolor="black")
        plt.title(f"Segment-wise MAE for {col} (RF Model)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"outputs/plots/segments_rf/segment_{col}_mae.png")
        plt.close()
#=========================end of segment analysis====================================================
        # ------------------ Feature Importance ------------------
        print("\n📊 Saving feature importance plot...")
        importances = best_rf.feature_importances_
        features = X_train.columns
        importance_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(10, 5))
        sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis")
        plt.title("Feature Importance")
        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        os.makedirs("outputs/plots", exist_ok=True)
        plt.savefig("outputs/plots/feature_importance_rf.png")
        plt.close()
#--------------------end of feture analysis-------------------------------------------------------
    os.makedirs("outputs/models", exist_ok=True)
    joblib.dump(best_rf, "outputs/models/best_rf_model.pkl")

    os.makedirs("outputs/metrics", exist_ok=True)
    with open("outputs/metrics/best_rf_metrics.json", "w") as f:
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

    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_test, y=test_preds, alpha=0.5)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Test: Actual vs Predicted")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig("outputs/plots/test_actual_vs_pred_rf.png")
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
    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig("outputs/plots/test_residual_rf.png")
    plt.close()

    print("🔍 Starting SHAP...")

    def run_fast_shap_analysis(model, X_train):
        try:
            os.environ["JOBLIB_MULTIPROCESSING"] = "0"  # ✅ Avoid SHAP loky deadlock
            print("\n⚙️ Running fast SHAP analysis with TreeExplainer...")

            # ✅ Use sampled subset for faster SHAP computation
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
            # ✅ Save top 5 SHAP features to JSON
            top_5_shap = shap_df.head(5).set_index("Feature")["Mean |SHAP|"].to_dict()
            os.makedirs("outputs/metrics", exist_ok=True)
            with open("outputs/metrics/rf_shap_importance.json", "w") as f:
                json.dump(top_5_shap, f, indent=4)
            print("\n💾 Saved top 5 SHAP features to outputs/metrics/rf_shap_importance.json")
        except Exception as e:
            print("⚠️ SHAP analysis failed:", str(e))

    
    run_fast_shap_analysis(best_rf, X_train)

    gap = round(r2 - test_r2, 4)
    if gap > 0.15 and test_r2 < 0.75:
        print("\n⚠️ Likely Overfitting Detected!")
        print(f"   - Train R² is {gap} higher than Test R²")
    elif gap < 0.05:
        print("\n✅ Model is well-generalized!")
    else:
        print("\nℹ️ Slight overfitting, may be acceptable depending on use case.")

    return best_rf, {"r2": r2, "mae": mae, "rmse": rmse}, {"r2": test_r2, "mae": test_mae, "rmse": test_rmse}

# 🔽 🔽 🔽 FUNCTION CALL — DO NOT REMOVE 🔽 🔽 🔽
X_train = pd.read_csv("data_processing/X_train.csv")
y_train = pd.read_csv("data_processing/Y_train.csv").squeeze()
X_test = pd.read_csv("data_processing/X_test.csv")
y_test = pd.read_csv("data_processing/Y_test.csv").squeeze()

rf_model, rf_train_metrics, rf_test_metrics = train_rf_model(X_train, y_train, X_test, y_test)
#===============================================================================================================
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

    if rf_r2_score < 0.5:
        fallback_time_budget += 60
    elif rf_r2_score < 0.75:
        fallback_time_budget += 30

    fallback_time_budget = min(fallback_time_budget, 300)
    print(f"⏳ Final FLAML Time Budget Set To: {fallback_time_budget} seconds")

    automl_settings = {
        "time_budget": fallback_time_budget,
        "metric": 'r2',
        "task": 'regression',
        "log_file_name": "flaml_fallback.log",
        "estimator_list": ["lgbm", "xgboost", "rf"],
        "seed": 42
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
        "rmse": round(mean_squared_error(y_train, train_preds, squared=False), 4)
    }

    test_metrics = {
        "r2": round(r2_score(y_test, test_preds), 4),
        "mae": round(mean_absolute_error(y_test, test_preds), 4),
        "rmse": round(mean_squared_error(y_test, test_preds, squared=False), 4)
    }

    os.makedirs("outputs/models", exist_ok=True)
    joblib.dump(automl, "outputs/models/fallback_flaml_model.pkl")

    print("\n📉 Fallback Model Performance:")
    print(f"   - Test R²:  {test_metrics['r2']}")
    print(f"   - MAE:      {test_metrics['mae']}")
    print(f"   - RMSE:     {test_metrics['rmse']}")

    os.makedirs("outputs/metrics", exist_ok=True)
    with open("outputs/metrics/fallback_flaml_metrics.json", "w") as f:
        json.dump({
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "training_time": training_time,
            "fallback_time_budget": fallback_time_budget,
            "best_estimator": str(automl.best_estimator)
        }, f, indent=4)

    return automl, train_metrics, test_metrics

# 🔥 Use the fallback if RF underperforms or overfits
print("\n🔎 Checking if fallback model is needed...")

rf_train_r2 = rf_train_metrics["r2"]
rf_test_r2 = rf_test_metrics["r2"]
gap = round(rf_train_r2 - rf_test_r2, 4)

if rf_test_r2 < 0.75 or (gap > 0.15 and rf_test_r2 < 0.75):
    print(f"\n⚠️ RF model underperformed or overfitting detected (gap = {gap})")
    fallback_model, fallback_train_metrics, fallback_test_metrics = train_flaml_fallback(
        X_train, y_train, X_test, y_test, rf_test_metrics
    )
    final_model = fallback_model
    used_model_type = "flaml_fallback"
    print("\n✅ Fallback model trained and selected.")
else:
    final_model = rf_model
    used_model_type = "random_forest"
    print("\n✅ RF model performed well. No fallback needed.")

