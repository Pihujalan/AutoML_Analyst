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
target_col = "target"
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

    if rf_r2_score =< 0.3:
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

run_full_analysis(final_model, X_train, y_train, X_test, y_test, model_name=used_model_type)
