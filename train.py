import pandas as pd
import numpy as np
import joblib
import math

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ==============================
# 1. LOAD DATA
# ==============================
df = pd.read_csv("salaries.csv")

print("\n====================")
print("INITIAL DATA")
print("====================")
print(df.shape)
print(df.head())

# ==============================
# 2. SELECT FEATURES
# ==============================
features = [
    "experience_level",
    "employment_type",
    "company_size",
    "remote_ratio",
    "job_title",
    "company_location"
]

target = "salary_in_usd"

df = df[features + [target]].copy()

# ==============================
# 3. CLEANING
# ==============================

# Remove duplicates
df = df.drop_duplicates()

# Remove outliers (tightened)
df = df[(df[target] > 15000) & (df[target] < 300000)]

# Fill missing categorical
for col in features:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\n====================")
print("AFTER CLEANING")
print("====================")
print(df.shape)

# ==============================
# 4. 🔥 BULLETPROOF ENCODING
# ==============================

encoders = {}

for col in features:
    df[col] = df[col].astype(str)  # force string
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features matrix
X = df[features]

# ==============================
# 5. TARGET TRANSFORMATION
# ==============================

# 🔥 VERY IMPORTANT (boost performance)
y = np.log1p(df[target])

print("\nFinal feature shape:", X.shape)
print("Feature types:\n", X.dtypes)

# ==============================
# 6. TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 7. MODEL TUNING
# ==============================
param_grid = {
    "max_depth": [5, 10, 15, 20, 25],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10]
}

search = RandomizedSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid,
    n_iter=20,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    random_state=42,
    error_score="raise"
)

search.fit(X_train, y_train)

model = search.best_estimator_

print("\nBest Parameters:", search.best_params_)

# ==============================
# 8. EVALUATION
# ==============================
y_pred_log = model.predict(X_test)

# Convert back from log
y_pred = np.expm1(y_pred_log)
y_test_real = np.expm1(y_test)

mae = mean_absolute_error(y_test_real, y_pred)
mse = mean_squared_error(y_test_real, y_pred)
rmse = math.sqrt(mse)
r2 = r2_score(y_test_real, y_pred)

print("\n====================")
print("MODEL PERFORMANCE")
print("====================")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2:   {r2:.4f}")

# ==============================
# 9. FEATURE IMPORTANCE
# ==============================
importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

print("\nTop Features:")
print(importance)

# ==============================
# 10. SAVE MODEL
# ==============================
joblib.dump(model, "decision_tree_model.pkl")
joblib.dump(encoders, "encoders.pkl")

print("\nModel saved successfully!")

# ==============================
# 11. SAVE METRICS
# ==============================
metrics = {
    "mae": mae,
    "rmse": rmse,
    "r2": r2
}

joblib.dump(metrics, "metrics.pkl")

# ==============================
# 12. SAVE CLEAN DATA
# ==============================
df.to_csv("cleaned_salaries.csv", index=False)