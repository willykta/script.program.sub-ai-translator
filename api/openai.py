import json
import urllib.request

import sys
import traceback

def call(prompt, model, api_key):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4
    }

    try:
        req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers)
        with urllib.request.urlopen(req) as res:
            response = json.loads(res.read().decode())
            return response["choices"][0]["message"]["content"]
    except Exception:
        print(f"[Sub-AI Translator] Request failed for model={model}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        raise
