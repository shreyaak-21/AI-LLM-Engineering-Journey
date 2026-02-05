# llm_client.py
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def generate_response(prompt, temperature=0.7, max_tokens=256, model="mistral"):
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "num_predict": max_tokens,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    return response.json()["response"]
