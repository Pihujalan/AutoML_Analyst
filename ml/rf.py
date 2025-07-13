import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 🔹 Load training data
X_train = pd.read_csv("mansi/X_train.csv")
y_train = pd.read_csv("mansi/Y_train.csv").squeeze()  # convert to Series
y_test = pd.read_csv("mansi/Y_test.csv").squeeze()  # convert to Series
X_test = pd.read_csv("mansi/X_test.csv")
# 🔹 Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True]
}

# 🔹 Create Random Forest model
rf = RandomForestRegressor(random_state=42)

# 🔹 RandomizedSearchCV for tuning
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=40,  # Try 40 different random combinations
    cv=5,  # 5-fold cross-validation
    verbose=2,  # Show progress
    n_jobs=-1,  # Use all CPU cores
    scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'],
    refit='r2',  # Select best based on R²
    random_state=42
)

# 🔹 Train the model
random_search.fit(X_train, y_train)
best_rf = random_search.best_estimator_

# 🔹 Predict on training set
train_preds = best_rf.predict(X_train)

# 🔹 Evaluate model
r2 = r2_score(y_train, train_preds)
mae = mean_absolute_error(y_train, train_preds)
rmse = mean_squared_error(y_train, train_preds, squared=False)

# 🔹 Print results
print("\n🎯 Best Hyperparameters Found:")
print(random_search.best_params_)

print("\n📊 Model Performance on Training Data:")
print(f"   - R² Score:  {round(r2, 4)}")
print(f"   - MAE:       {round(mae, 4)}")
print(f"   - RMSE:      {round(rmse, 4)}")
