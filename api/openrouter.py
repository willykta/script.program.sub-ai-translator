import json
import time
import urllib.request
import urllib.error


def call(prompt, model, api_key, logger=print):
    """
    Make a chat completion request to OpenRouter using OpenAI-compatible schema.

    Args:
        prompt (str): The user prompt to send.
        model (str): The OpenRouter model id (e.g. 'openai/gpt-4o-mini').
        api_key (str): OpenRouter API key.
        logger (callable): Logging function for debug output (default: print).

    Returns:
        str: The assistant message content.

    Raises:
        urllib.error.HTTPError: On non-2xx HTTP responses.
        Exception: On other unexpected errors.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        # Optional but useful for OpenRouter analytics; safe to omit if not available:
        # "HTTP-Referer": "https://example.com",
        # "X-Title": "Sub-AI Translator",
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
    }

    logger(f"[OPENROUTER] Request to model={model}, prompt={repr(prompt[:60])}...")

    try:
        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
        start = time.time()
        with urllib.request.urlopen(req, timeout=10) as res:
            duration = time.time() - start
            payload = res.read().decode("utf-8")
            response = json.loads(payload)

            # OpenAI-compatible response shape
            content = response["choices"][0]["message"]["content"]
            logger(f"[OPENROUTER] Response in {duration:.2f}s: {repr(content[:60])}...")
            return content

    except urllib.error.HTTPError as e:
        code = e.code
        # Read body to include in logs for diagnostics
        try:
            body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            body = ""
        logger(f"[OPENROUTER] HTTPError {code}: {body}")
        # Attach a neutral attribute for upstream inspection (pattern used in gemini_api)
        e._http_status = code
        raise e
    except Exception as e:
        logger(f"[OPENROUTER] Unexpected error: {type(e).__name__}: {str(e)}")
        raise