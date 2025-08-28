import json
import time
import urllib.request
import urllib.error
import xbmc

# Try to import urllib3 for connection pooling
try:
    import urllib3
    from connection_pool import connection_pool_manager
    HAS_URLLIB3 = True
except ImportError:
    HAS_URLLIB3 = False
    xbmc.log("[OPENROUTER] urllib3 not available, using fallback implementation", xbmc.LOGWARNING)

def call(prompt, model, api_key, logger=None):
    """
    Make a chat completion request to OpenRouter using OpenAI-compatible schema.

    Args:
        prompt (str): The user prompt to send.
        model (str): The OpenRouter model id (e.g. 'openai/gpt-4o-mini').
        api_key (str): OpenRouter API key.
        logger (callable): Logging function for debug output (default: xbmc.log).

    Returns:
        str: The assistant message content.

    Raises:
        urllib.error.HTTPError: On non-2xx HTTP responses.
        Exception: On other unexpected errors.
    """
    # Use xbmc.log as default logger if none provided
    if logger is None:
        logger = lambda msg: xbmc.log(msg, xbmc.LOGDEBUG)
    
    # Try to use connection pooling if available
    if HAS_URLLIB3:
        return _call_with_pooling(prompt, model, api_key, logger)
    else:
        # Fallback to original implementation
        return _call_fallback(prompt, model, api_key, logger)

def _call_with_pooling(prompt, model, api_key, logger):
    """
    Make a chat completion request using connection pooling.
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
        # Get connection pool
        pool = connection_pool_manager.get_pool()
        if pool is None:
            # Fallback to original implementation if pooling fails
            logger("[OPENROUTER] Connection pooling failed, using fallback")
            return _call_fallback(prompt, model, api_key, logger)
        
        start = time.time()
        # Use connection pool for request with improved timeout handling
        response = pool.request(
            'POST',
            url,
            body=json.dumps(data).encode('utf-8'),
            headers=headers,
            timeout=urllib3.Timeout(connect=2.0, read=15.0)  # Match connection pool timeouts
        )
        
        duration = time.time() - start
        
        if response.status >= 400:
            # Handle HTTP errors
            body = response.data.decode('utf-8', errors='ignore')
            logger(f"[OPENROUTER] HTTPError {response.status}: {body}")
            # Create exception similar to urllib.error.HTTPError for compatibility
            from urllib.error import HTTPError
            e = HTTPError(url, response.status, body, response.headers, None)
            e._http_status = response.status
            raise e
        
        payload = response.data.decode('utf-8')
        response_data = json.loads(payload)

        # OpenAI-compatible response shape
        content = response_data["choices"][0]["message"]["content"]
        logger(f"[OPENROUTER] Response in {duration:.2f}s: {repr(content[:60])}...")
        return content
        
    except Exception as e:
        logger(f"[OPENROUTER] Unexpected error: {type(e).__name__}: {str(e)}")
        raise

def _call_fallback(prompt, model, api_key, logger):
    """
    Fallback to original urllib.request implementation.
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

    logger(f"[OPENROUTER] Fallback request to model={model}, prompt={repr(prompt[:60])}...")

    try:
        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
        start = time.time()
        with urllib.request.urlopen(req, timeout=10) as res:
            duration = time.time() - start
            payload = res.read().decode("utf-8")
            response = json.loads(payload)

            # OpenAI-compatible response shape
            content = response["choices"][0]["message"]["content"]
            logger(f"[OPENROUTER] Fallback response in {duration:.2f}s: {repr(content[:60])}...")
            return content

    except urllib.error.HTTPError as e:
        code = e.code
        # Read body to include in logs for diagnostics
        try:
            body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            body = ""
        logger(f"[OPENROUTER] Fallback HTTPError {code}: {body}")
        # Attach a neutral attribute for upstream inspection (pattern used in gemini_api)
        e._http_status = code
        raise e
    except Exception as e:
        logger(f"[OPENROUTER] Fallback unexpected error: {type(e).__name__}: {str(e)}")
        raise