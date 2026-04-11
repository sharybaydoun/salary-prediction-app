from fastapi import FastAPI
import pandas as pd
import joblib
import numpy as np
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# ========================
# LOAD MODEL + ENCODERS
# ========================
model = joblib.load("decision_tree_model.pkl")
encoders = joblib.load("encoders.pkl")

# ========================
# SUPABASE
# ========================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ========================
# HOME
# ========================
@app.get("/")
def home():
    return {"message": "API is running"}

# ========================
# PREDICT
# ========================
@app.get("/predict")
def predict(
    experience_level: str,
    employment_type: str,
    company_size: str,
    remote_ratio: int,
    job_title: str = "Data Scientist",
    company_location: str = "US"
):
    try:
        # Create input
        input_data = pd.DataFrame([{
            "experience_level": experience_level,
            "employment_type": employment_type,
            "company_size": company_size,
            "remote_ratio": remote_ratio,
            "job_title": job_title,
            "company_location": company_location
        }])

        # 🔥 Apply SAME encoding as training
        for col in input_data.columns:
            input_data[col] = input_data[col].astype(str)

            if col in encoders:
                le = encoders[col]

                if input_data[col].iloc[0] not in le.classes_:
                    input_data[col] = le.transform([le.classes_[0]])
                else:
                    input_data[col] = le.transform(input_data[col])

        # Predict (log salary)
        prediction_log = model.predict(input_data)[0]

        # Convert to real salary
        prediction = np.expm1(prediction_log)

        # Save to Supabase
        try:
            supabase.table("predictions").insert({
                "experience_level": experience_level,
                "employment_type": employment_type,
                "company_size": company_size,
                "remote_ratio": remote_ratio,
                "job_title": job_title,
                "company_location": company_location,
                "salary": float(prediction)
            }).execute()
        except Exception as e:
            print("Supabase error:", e)

        return {"salary": round(float(prediction), 2)}

    except Exception as e:
        return {"error": str(e)}