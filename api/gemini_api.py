import json
import urllib.request
import urllib.error
import time

def call(prompt, model, api_key, logger=print):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }]
    }

    logger(f"[GEMINI] Request to model={model}, prompt={repr(prompt[:60])}...")

    try:
        req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers)
        start = time.time()
        with urllib.request.urlopen(req, timeout=10) as res:
            duration = time.time() - start
            response = json.loads(res.read().decode())
            text = response["candidates"][0]["content"]["parts"][0]["text"]
            logger(f"[GEMINI] Response in {duration:.2f}s: {repr(text[:60])}...")
            return text
    except urllib.error.HTTPError as e:
        code = e.code
        body = e.read().decode("utf-8", errors="ignore")
        logger(f"[GEMINI] HTTPError {code}: {body}")
        e._http_status = code  # zamiast niedozwalanej .status
        raise e
    except Exception as e:
        logger(f"[GEMINI] Unexpected error: {type(e).__name__}: {str(e)}")
        raise
