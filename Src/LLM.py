import requests

def generate_response(prompt):

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": False     # get answer completely before returning, true means it return each word in each lines

        }
    )

    return response.json()["response"]