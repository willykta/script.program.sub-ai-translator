import json
import urllib.request
import urllib.error
import time
import xbmc

# Try to import urllib3 for connection pooling
try:
    import urllib3
    from connection_pool import connection_pool_manager
    HAS_URLLIB3 = True
except ImportError:
    HAS_URLLIB3 = False
    xbmc.log("[GEMINI] urllib3 not available, using fallback implementation", xbmc.LOGWARNING)

def get_api_version_for_model(model: str) -> str:
    return "v1" if model.startswith("gemini-2.") else "v1beta"

def call(prompt, model, api_key, logger=None):
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
    api_version = get_api_version_for_model(model)
    url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }]
    }

    logger(f"[GEMINI] Request to model={model}, prompt={repr(prompt[:60])}...")

    try:
        # Get connection pool
        pool = connection_pool_manager.get_pool()
        if pool is None:
            # Fallback to original implementation if pooling fails
            logger("[GEMINI] Connection pooling failed, using fallback")
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
            logger(f"[GEMINI] HTTPError {response.status}: {body}")
            # Create exception similar to urllib.error.HTTPError for compatibility
            from urllib.error import HTTPError
            e = HTTPError(url, response.status, body, response.headers, None)
            e._http_status = response.status
            raise e
        
        response_data = json.loads(response.data.decode())
        text = response_data["candidates"][0]["content"]["parts"][0]["text"]
        logger(f"[GEMINI] Response in {duration:.2f}s: {repr(text[:60])}...")
        return text
        
    except Exception as e:
        logger(f"[GEMINI] Unexpected error: {type(e).__name__}: {str(e)}")
        raise

def _call_fallback(prompt, model, api_key, logger):
    """
    Fallback to original urllib.request implementation.
    """
    api_version = get_api_version_for_model(model)
    url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }]
    }

    logger(f"[GEMINI] Fallback request to model={model}, prompt={repr(prompt[:60])}...")

    try:
        req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers)
        start = time.time()
        with urllib.request.urlopen(req, timeout=10) as res:
            duration = time.time() - start
            response = json.loads(res.read().decode())
            text = response["candidates"][0]["content"]["parts"][0]["text"]
            logger(f"[GEMINI] Fallback response in {duration:.2f}s: {repr(text[:60])}...")
            return text
    except urllib.error.HTTPError as e:
        code = e.code
        body = e.read().decode("utf-8", errors="ignore")
        logger(f"[GEMINI] Fallback HTTPError {code}: {body}")
        e._http_status = code  # zamiast niedozwalanej .status
        raise e
    except Exception as e:
        logger(f"[GEMINI] Fallback unexpected error: {type(e).__name__}: {str(e)}")
        raise
