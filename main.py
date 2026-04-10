from fastapi import FastAPI
import pandas as pd
import joblib
from supabase import create_client
import os

from dotenv import load_dotenv
import os

load_dotenv()
# -----------------------
# INIT APP
# -----------------------
app = FastAPI()

# -----------------------
# LOAD MODEL
# -----------------------
model = joblib.load("decision_tree_model.pkl")
columns = joblib.load("model_columns.pkl")

# -----------------------
# SUPABASE CONNECTION
# -----------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------
# ROUTES
# -----------------------
@app.get("/")
def home():
    return {"message": "API is running"}

@app.get("/predict")
def predict(
    experience_level: str,
    employment_type: str,
    company_size: str,
    remote_ratio: int
):
    # Create input
    input_data = pd.DataFrame([{
        "experience_level": experience_level,
        "employment_type": employment_type,
        "company_size": company_size,
        "remote_ratio": remote_ratio
    }])

    # Encode
    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

    # Predict
    prediction = model.predict(input_encoded)[0]

    # -----------------------
    # SAVE TO SUPABASE
    # -----------------------
    try:
        supabase.table("predictions").insert({
            "experience_level": experience_level,
            "employment_type": employment_type,
            "company_size": company_size,
            "remote_ratio": remote_ratio,
            "salary": float(prediction)
        }).execute()
    except Exception as e:
        print("Supabase insert error:", e)

    # Return result
    return {"salary": round(prediction, 2)}