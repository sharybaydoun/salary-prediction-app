import requests

def explain_salary(salary):
    prompt = f"""
    A data science salary prediction is {salary} USD.

    Explain:
    - is this high or low
    - what factors affect it
    - what this means for a job seeker

    Keep it short, simple, and insightful.
    """

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen3.5:4b",
                "prompt": prompt,
                "stream": False
            }
        )

        result = response.json()
        return result["response"]

    except Exception as e:
        return f"LLM error: {e}"