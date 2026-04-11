# 💼 Salary Prediction App

A full-stack machine learning application that predicts data science salaries, visualizes trends, and provides AI-generated explanations.

---

## 🚀 Project Overview

This project combines machine learning, data visualization, backend APIs, and AI explanations into a single interactive application.

It includes:

- 📊 Streamlit dashboard for data exploration and predictions  
- 🤖 Decision Tree model with optimized hyperparameters  
- 🌐 FastAPI backend for serving predictions  
- 🧠 Local LLM using Ollama for AI explanations  
- 🗄️ Supabase database to store prediction history  

---

## 📁 Project Structure
salary-prediction-app/
│
├── app.py                  # Streamlit dashboard (EDA + prediction UI + LLM)
├── main.py                 # FastAPI backend
├── client.py               # API testing script
├── llm.py                  # LLM explanation module
├── test_llm.py             # LLM testing script
├── salaries.csv            # Original dataset
├── cleaned_salaries.csv    # Cleaned dataset
├── decision_tree_model.pkl # Trained model
├── encoders.pkl            # Label encoders
├── metrics.pkl             # Model performance
├── requirements.txt        # Dependencies
└── README.md

---

## ⚙️ Features

### 📊 Dashboard (Streamlit)
- Salary distribution visualization  
- Salary vs experience analysis  
- Salary vs remote work analysis  
- Average salary trends by experience  
- Prediction interface with user inputs  

---

### 🤖 Salary Prediction

Predict salary using:

- Experience level (EN, MI, SE, EX)  
- Employment type (FT, PT, CT, FL)  
- Company size (S, M, L)  
- Remote ratio (0–100)  
- Job title (Data Scientist, Data Engineer, Data Analyst)  
- Company location (US, UK, CA, IN)  

---

### 🧠 AI Explanation (Ollama)

- Uses a local LLM via Ollama (`phi3:mini`)  
- Generates short, human-readable explanations  
- Explains why the predicted salary makes sense  

---

### 🗄️ Database Storage (Supabase)

- Stores prediction inputs and outputs  
- Displays prediction history in dashboard  
- Enables simple data persistence  

---

## 🧠 Model Details

- Model: Decision Tree Regressor  
- Hyperparameter tuning: RandomizedSearchCV  
- Target transformation: Log transformation (`log1p`)  
- Encoding: Label Encoding (consistent with training)  

### Data Cleaning
- Removed duplicates  
- Removed outliers:
  - salary < 15,000  
  - salary > 300,000  
- Filled missing categorical values  

---

## 📊 Model Performance

- **MAE**: 26,907 USD  
- **RMSE**: 38,415 USD  
- **R² Score**: 0.536   

---

## ▶️ How to Run

### 1️⃣ Install dependencies
pip install -r requirements.txt
---

### 2️⃣ Train the model
python train.py

---

### 3️⃣ Run FastAPI backend
uvicorn main:app –reload

---

### 4️⃣ Run Streamlit dashboard

streamlit run app.py

---

### 5️⃣ Run Ollama (for AI explanations)
ollama run phi3:mini

---

## 🔌 API Endpoint

### `/predict`

**GET parameters:**

- `experience_level` → EN, MI, SE, EX  
- `employment_type` → FT, PT, CT, FL  
- `company_size` → S, M, L  
- `remote_ratio` → 0–100  
- `job_title` → Data Scientist, Data Engineer, Data Analyst  
- `company_location` → US, UK, CA, IN  

### Example:
http://127.0.0.1:8000/predict?experience_level=SE&employment_type=FT&company_size=L&remote_ratio=100&job_title=Data Scientist&company_location=US

---

## 🤖 LLM Usage

Used automatically inside the app, or manually:

```python
from llm import explain_salary

print(explain_salary(120000, "SE", "FT", "L", 100))

🗄️ Database Setup

Create a .env file:
SUPABASE_URL=your_url
SUPABASE_KEY=your_key

Table: predictions

Columns:
	•	experience_level
	•	employment_type
	•	company_size
	•	remote_ratio
	•	job_title
	•	company_location
	•	salary
	•	created_at

⚠️ Notes
	•	FastAPI must be running before predictions
	•	Ollama must be running for AI explanations
	•	.env file is not included in GitHub
	•	Supabase RLS must be disabled or configured

📌 Future Improvements
	•	Improve model (Random Forest, XGBoost)
	•	Deploy application online
	•	Enhance UI/UX
	•	Add authentication system
	•	Add more features (education, company type, etc.)

🎯 Summary

This project demonstrates:
	•	Data analysis and visualization
	•	Machine learning modeling
	•	Backend API development
	•	Database integration
	•	AI-powered explanations
