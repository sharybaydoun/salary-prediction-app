import requests

url = "http://127.0.0.1:8000/predict"

params = {
    "experience_level": "SE",
    "employment_type": "FT",
    "company_size": "L",
    "remote_ratio": 100
}

response = requests.get(url, params=params)

print("Response:")
print(response.json())