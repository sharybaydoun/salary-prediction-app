import requests

url = "http://127.0.0.1:8000/predict"

params = {
    "experience_level": "SE",
    "employment_type": "FT",
    "company_size": "L",
    "remote_ratio": 100,
    "job_title": "Data Scientist",      # ✅ ADD THIS
    "company_location": "US"            # ✅ ADD THIS
}

response = requests.get(url, params=params)

print("Status Code:", response.status_code)
print("Raw Response:", response.text)

# Safe JSON parsing
try:
    print("Parsed JSON:", response.json())
except:
    print("❌ API did not return JSON")