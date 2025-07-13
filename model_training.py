# Step 1: Import required libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib  # Used to save the trained model
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the training and testing datasets
print("📥 Loading training and testing datasets...")
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
Y_train = pd.read_csv("Y_train.csv")
Y_test = pd.read_csv("Y_test.csv")
print("✅ Data loaded successfully.")
print("🔹 Y_train shape:", Y_train.shape)
print(Y_train.head()) 

# Step 3: Initialize and train the Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(
    n_estimators=100,       # Number of boosting stages (trees)
    learning_rate=0.1,      # How much each tree corrects the previous one
    max_depth=3,            # Depth of each tree
    random_state=42         # Reproducibility
)
print("\n🚀 Training the Gradient Boosting Regressor...")
gb_model.fit(X_train, Y_train.values.ravel())
print("✅ Model training complete.")

# Step 4: Save the trained model
joblib.dump(gb_model, "trained_gbm_model.pkl")
print("💾 Model saved as 'trained_gbm_model.pkl'.")

# Step 5: Evaluate the model on the test set
print("\n📊 Evaluating model performance...")
y_pred = gb_model.predict(X_test)

# Calculate regression metrics
mae = mean_absolute_error(Y_test, y_pred)
mse = mean_squared_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

# Print metrics
print("📈 Regression Evaluation Results:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

pred_df = pd.DataFrame({
    "Actual": Y_test.values.ravel(),
    "Predicted": y_pred
})
pred_df.to_csv("gbm_predictions.csv", index=False)
print("Predictions saved to gbm_predictions.csv")

#scatter plot:actual vs predicted

plt.figure(figsize=(8,6))
sns.scatterplot(x=Y_test.values.ravel(), y=y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values (GBM Regression)")
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/actual_vs_predicted_gbm.png")
plt.show()

#Residual Plot
residuals = Y_test.values.ravel() - y_pred
plt.figure(figsize=(8,6))
sns.histplot(residuals, bins=30, kde=True)
plt.xlabel("Residuals")
plt.title("Distribution of Residuals")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/residuals_gbm.png")
plt.show()