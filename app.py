import streamlit as st
import pandas as pd
import requests
import seaborn as sns
import matplotlib.pyplot as plt
from supabase import create_client
from llm import explain_salary
import os

from dotenv import load_dotenv
import os

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
# LOAD DATA (FOR EDA ONLY)
# ========================
df = pd.read_csv("salaries.csv")

# ========================
# CLEAN DATA (LIGHT CLEAN)
# ========================
df = df[(df["salary_in_usd"] > 10000) & (df["salary_in_usd"] < 500000)]

# ========================
# 📊 EDA SECTION
# ========================
st.header("📊 Exploratory Data Analysis")

# ------------------------
# Salary Distribution
# ------------------------
st.subheader("Salary Distribution")

fig, ax = plt.subplots()
sns.histplot(df["salary_in_usd"], bins=30, kde=True, ax=ax)
st.pyplot(fig)

st.write("""
💡 Insight:
The salary distribution is right-skewed.
Most data science roles cluster between 50k–150k USD,
but a small number of senior roles push salaries much higher.
""")

# ------------------------
# Experience vs Salary
# ------------------------
st.subheader("📊 Salary by Experience")

fig, ax = plt.subplots()
sns.boxplot(x="experience_level", y="salary_in_usd", data=df, ax=ax)
st.pyplot(fig)

st.write("""
💡 Insight:
Salaries increase significantly with experience.
Senior and executive roles dominate the high salary range.
""")

# ------------------------
# Remote work
# ------------------------
st.subheader("📊 Salary by Remote Work")

fig, ax = plt.subplots()
sns.boxplot(x="remote_ratio", y="salary_in_usd", data=df, ax=ax)
st.pyplot(fig)

st.write("""
💡 Insight:
Higher remote ratios often correlate with higher salaries.
Global companies tend to offer better compensation.
""")

# ------------------------
# Average salary by experience
# ------------------------
st.subheader("📊 Average Salary by Experience")

avg_salary = df.groupby("experience_level")["salary_in_usd"].mean().sort_values()
st.bar_chart(avg_salary)

st.write("""
💡 Insight:
Clear progression:
Entry → Mid → Senior → Executive.
Experience is the strongest driver of salary.
""")

# ========================
# 🤖 PREDICTION SECTION
# ========================
st.header("🤖 Salary Prediction")

experience = st.selectbox("Experience Level", ["EN", "MI", "SE", "EX"])
employment = st.selectbox("Employment Type", ["FT", "PT", "CT", "FL"])
company = st.selectbox("Company Size", ["S", "M", "L"])
remote = st.slider("Remote Ratio", 0, 100, 50)

if st.button("Predict Salary"):
    try:
        response = requests.get(
            "http://127.0.0.1:8000/predict",
            params={
                "experience_level": experience,
                "employment_type": employment,
                "company_size": company,
                "remote_ratio": remote
            }
        )

        result = response.json()
        salary = result["salary"]

        st.success(f"💰 Predicted Salary: {salary} USD")

        # ------------------------
        # LLM EXPLANATION
        # ------------------------
        st.subheader("🤖 AI Explanation")

        explanation = explain_salary(salary)
        st.write(explanation)

    except Exception as e:
        st.error(f"API error: {e}")

# ========================
# 📜 HISTORY FROM SUPABASE
# ========================
st.header("📜 Prediction History")

try:
    response = supabase.table("predictions").select("*").execute()
    data = response.data

    if data:
        history_df = pd.DataFrame(data)
        st.dataframe(history_df)
    else:
        st.write("No predictions yet.")

except Exception as e:
    st.error(f"Error loading history: {e}")

# ========================
# 📈 MODEL PERFORMANCE
# ========================
st.header("📈 Model Performance")

st.write("""
MAE: ~15000 USD  
RMSE: ~25000 USD  
R²: ~0.75  

💡 Insight:
The model captures general salary trends well,
but extreme salaries remain harder to predict.
""")