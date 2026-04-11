import streamlit as st
import pandas as pd
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from supabase import create_client
from llm import explain_salary
import os
from dotenv import load_dotenv

load_dotenv()

# ========================
# SUPABASE CONFIG
# ========================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ========================
# TITLE
# ========================
st.title("💼 Data Science Salary Dashboard")
st.write("Understand salary trends and predict salaries using AI.")

# ========================
# LOAD DATA
# ========================
df = pd.read_csv("salaries.csv")
df = df[(df["salary_in_usd"] > 10000) & (df["salary_in_usd"] < 500000)]

# ========================
# EDA
# ========================
st.header("📊 Exploratory Data Analysis")

fig, ax = plt.subplots()
sns.histplot(df["salary_in_usd"], bins=30, kde=True, ax=ax)
st.pyplot(fig)

fig, ax = plt.subplots()
sns.boxplot(x="experience_level", y="salary_in_usd", data=df, ax=ax)
st.pyplot(fig)

fig, ax = plt.subplots()
sns.boxplot(x="remote_ratio", y="salary_in_usd", data=df, ax=ax)
st.pyplot(fig)

avg_salary = df.groupby("experience_level")["salary_in_usd"].mean()
st.bar_chart(avg_salary)

# ========================
# PREDICTION
# ========================
st.header("🤖 Salary Prediction")

experience = st.selectbox("Experience Level", ["EN", "MI", "SE", "EX"])
employment = st.selectbox("Employment Type", ["FT", "PT", "CT", "FL"])
company = st.selectbox("Company Size", ["S", "M", "L"])
remote = st.slider("Remote Ratio", 0, 100, 50)

# 👉 ADD THESE (VERY IMPORTANT)
job_title = st.selectbox("Job Title", ["Data Scientist", "Data Engineer", "Data Analyst"])
location = st.selectbox("Company Location", ["US", "UK", "CA", "IN"])

if st.button("Predict Salary"):
    try:
        response = requests.get(
            "http://127.0.0.1:8000/predict",
            params={
                "experience_level": experience,
                "employment_type": employment,
                "company_size": company,
                "remote_ratio": remote,
                "job_title": job_title,
                "company_location": location
            }
        )

        result = response.json()

        if "error" in result:
            st.error(result["error"])
        else:
            salary = result["salary"]

            st.success(f"💰 Predicted Salary: {salary:,.2f} USD")

            # LLM explanation
            with st.spinner("Generating AI explanation..."):
                explanation = explain_salary(
                    salary,
                    experience,
                    employment,
                    company,
                    remote
                )
            st.write(explanation)

    except Exception as e:
        st.error(f"API error: {e}")

# ========================
# HISTORY
# ========================
st.header("📜 Prediction History")

try:
    data = supabase.table("predictions").select("*").execute().data
    if data:
        st.dataframe(pd.DataFrame(data))
    else:
        st.write("No predictions yet.")
except Exception as e:
    st.error(e)

# ========================
# METRICS
# ========================
st.header("📈 Model Performance")

try:
    metrics = joblib.load("metrics.pkl")
    st.write(f"""
MAE: {metrics['mae']:.2f}  
RMSE: {metrics['rmse']:.2f}  
R²: {metrics['r2']:.4f}
""")
except:
    st.error("Run train.py first")