import json
import urllib.request

def call(prompt, model, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }]
    }

    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers)
    with urllib.request.urlopen(req) as res:
        response = json.loads(res.read().decode())
        return response["candidates"][0]["content"]["parts"][0]["text"]
