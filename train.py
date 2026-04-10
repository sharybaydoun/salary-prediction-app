import pandas as pd
import numpy as np
import joblib
import math

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    "remote_ratio"
]

target = "salary_in_usd"

df = df[features + [target]].copy()

# ==============================
# 3. DATA CLEANING
# ==============================

# Remove duplicates
df = df.drop_duplicates()

# Remove unrealistic salaries (outliers)
df = df[(df[target] > 10000) & (df[target] < 500000)]

# Fill missing values
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in df.select_dtypes(include=["int64", "float64"]).columns:
    df[col] = df[col].fillna(df[col].median())

print("\n====================")
print("AFTER CLEANING")
print("====================")
print(df.shape)

# ==============================
# 4. FEATURE ENGINEERING
# ==============================

# Encode categorical features
X = pd.get_dummies(df[features], drop_first=False)
y = df[target]

print("\nEncoded shape:", X.shape)

# ==============================
# 5. TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 6. TRAIN MODEL
# ==============================
model = DecisionTreeRegressor(
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

model.fit(X_train, y_train)

# ==============================
# 7. EVALUATION
# ==============================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n====================")
print("MODEL PERFORMANCE")
print("====================")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2:   {r2:.4f}")

# ==============================
# 8. FEATURE IMPORTANCE
# ==============================
importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

print("\nTop Features:")
print(importance.head(10))

# ==============================
# 9. SAVE MODEL
# ==============================
joblib.dump(model, "decision_tree_model.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("\nModel saved successfully!")

# ==============================
# 10. SAVE CLEAN DATA (OPTIONAL)
# ==============================
df.to_csv("cleaned_salaries.csv", index=False)