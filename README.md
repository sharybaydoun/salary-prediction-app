
💼 Salary Prediction App

A full-stack machine learning application that predicts data science salaries, visualizes trends, and provides AI-generated explanations.


🚀 Project Overview

This project combines machine learning, interactive visualization, and AI explanations into one application.

It includes:
	•	📊 Streamlit dashboard (in app.py) for data exploration and predictions
	•	🤖 Decision Tree model for salary prediction
	•	🌐 FastAPI backend (main.py) serving predictions
	•	🧠 Local LLM using Ollama for explanations
	•	🗄️ Database using Supabase to store predictions

salary-prediction-app/
│
├── app.py                  # Streamlit dashboard + EDA + prediction UI
├── main.py                 # FastAPI backend
├── client.py               # API testing script
├── llm.py                  # LLM explanation module
├── test_llm.py             # LLM testing
├── salaries.csv            # Dataset
├── cleaned_salaries.csv    # Cleaned dataset
├── decision_tree_model.pkl # Trained model
├── model_columns.pkl       # Encoded columns
├── requirements.txt        # Dependencies
└── README.md

⚙️ Features

📊 Dashboard (Streamlit)
	•	Salary distribution visualization
	•	Salary vs experience analysis
	•	Salary vs remote work
	•	Average salary trends

🤖 Salary Prediction

Predict salary using:
	•	Experience level (EN MI SE EX)
	•	Employment type (FT PT CT FL)
	•	Company size (S M L)
	•	Remote ratio (0–100)

🧠 AI Explanation
	•	Uses local LLM via Ollama
	•	Generates short explanations for predicted salaries

🗄️ Database Storage
	•	Saves predictions to Supabase
	•	Displays prediction history in dashboard

⸻

🧠 Model Details
	•	Model: Decision Tree Regressor
	•	Max depth: 12
	•	Min samples split: 10
	•	Min samples leaf: 5
	•	Encoding: One-hot encoding
	•	Outliers removed:
	•	salary < 10,000
	•	salary > 500,000

▶️ How to Run

1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Train the model
python app.py

3️⃣ Run FastAPI backend
uvicorn main:app --reload

4️⃣ Run Streamlit dashboard
streamlit run app.py

5️⃣ (Optional) Run Ollama
ollama run qwen3.5:4b

🔌 API Endpoint

/predict

GET parameters:
	•	experience_level → EN MI SE EX
	•	employment_type → FT PT CT FL
	•	company_size → S M L
	•	remote_ratio → 0–100

Example:
http://127.0.0.1:8000/predict?experience_level=SE&employment_type=FT&company_size=L&remote_ratio=100

🤖 LLM Usage

Inside the app automatically, or manually:
from llm import explain_salary
print(explain_salary(120000))

🗄️ Database Setup

Create .env file:
SUPABASE_URL=your_url
SUPABASE_KEY=your_key

Table name:
predictions

📊 Model Performance
	•	MAE: ~15,000 USD
	•	RMSE: ~25,000 USD
	•	R²: ~0.75

⸻

⚠️ Notes
	•	FastAPI must be running before prediction
	•	Ollama must be running for explanations
	•	.env is not pushed to GitHub

⸻

📌 Future Improvements
	•	Improve model (Random Forest / XGBoost)
	•	Deploy online
	•	Enhance UI
	•	Add authentication
