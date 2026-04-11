import requests

def explain_salary(salary, experience, employment, company, remote):
    prompt = f"""
    A data science salary of {salary} USD was predicted with:
    - Experience level: {experience}
    - Employment type: {employment}
    - Company size: {company}
    - Remote ratio: {remote}%

    Explain briefly:
    - why this salary makes sense
    - how these factors influence it
    - what it means for a job seeker

    Keep it short (3-4 sentences), clear, and specific.
    Do not use headings or markdown.
    """

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3:mini",
                "prompt": prompt,
                "stream": False
            }
        )

        result = response.json()
        return result["response"]

    except Exception as e:
        return f"LLM error: {e}"