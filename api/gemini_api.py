import json
import urllib.request
import urllib.error
import time

def call(prompt, model, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }]
    }

    print(f"[GEMINI] Request to model={model}, prompt={repr(prompt[:60])}...")

    try:
        req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers)
        start = time.time()
        with urllib.request.urlopen(req, timeout=30) as res:
            duration = time.time() - start
            response = json.loads(res.read().decode())
            text = response["candidates"][0]["content"]["parts"][0]["text"]
            print(f"[GEMINI] Response in {duration:.2f}s: {repr(text[:60])}...")
            return text
    except urllib.error.HTTPError as e:
        status = e.code
        body = e.read().decode("utf-8", errors="ignore")
        print(f"[GEMINI] HTTPError {status}: {body}")
        e.status = status
        raise e
    except Exception as e:
        print(f"[GEMINI] Unexpected error: {type(e).__name__}: {str(e)}")
        raise
