import json
import urllib.request
import urllib.error
import time
import sys
import traceback

def call(prompt, model, api_key):
    """
    Make a chat completion request to OpenAI API.
    
    Args:
        prompt (str): The user prompt to send.
        model (str): The OpenAI model id (e.g. 'gpt-4o-mini', 'gpt-5').
        api_key (str): OpenAI API key.
        
    Returns:
        str: The assistant message content.
        
    Raises:
        urllib.error.HTTPError: On non-2xx HTTP responses.
        Exception: On other unexpected errors.
    """
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
    
    # Log the request for debugging (using print for compatibility)
    print(f"[OPENAI] Request to model={model}, prompt={repr(prompt[:60])}...", file=sys.stderr)
    
    try:
        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
        start = time.time()
        # Add timeout handling (30 seconds)
        with urllib.request.urlopen(req, timeout=30) as res:
            duration = time.time() - start
            payload = res.read().decode("utf-8")
            response = json.loads(payload)
            
            # OpenAI-compatible response shape - works with all OpenAI models including GPT-5
            content = response["choices"][0]["message"]["content"]
            print(f"[OPENAI] Response in {duration:.2f}s: {repr(content[:60])}...", file=sys.stderr)
            return content
            
    except urllib.error.HTTPError as e:
        code = e.code
        # Read body to include in logs for diagnostics
        try:
            body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            body = ""
            
        print(f"[OPENAI] HTTPError {code}: {body}", file=sys.stderr)
        
        # Add specific error handling for common OpenAI API errors
        if code == 429:
            print("[OPENAI] Rate limit exceeded. Please wait before sending another request.", file=sys.stderr)
        elif code == 401:
            print("[OPENAI] Authentication failed. Please check your API key.", file=sys.stderr)
        elif code == 400:
            print("[OPENAI] Bad request. Please check your request parameters.", file=sys.stderr)
        elif code >= 500:
            print(f"[OPENAI] Server error ({code}). Please try again later.", file=sys.stderr)
            
        # Attach a neutral attribute for upstream inspection
        e._http_status = code
        raise e
    except Exception as e:
        print(f"[OPENAI] Unexpected error: {type(e).__name__}: {str(e)}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        raise
