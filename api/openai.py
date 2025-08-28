import json
import urllib.request
import urllib.error
import time
import sys
import traceback
import xbmc

# Try to import urllib3 for connection pooling
try:
    import urllib3
    from core.connection_pool import connection_pool_manager
    HAS_URLLIB3 = True
except ImportError:
    HAS_URLLIB3 = False
    xbmc.log("[OPENAI] urllib3 not available, using fallback implementation", xbmc.LOGWARNING)

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
    # Try to use connection pooling if available
    if HAS_URLLIB3:
        return _call_with_pooling(prompt, model, api_key)
    else:
        # Fallback to original implementation
        return _call_fallback(prompt, model, api_key)

def _call_with_pooling(prompt, model, api_key):
    """
    Make a chat completion request using connection pooling.
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
    
    xbmc.log(f"[OPENAI] Request to model={model}, prompt={repr(prompt[:60])}...", xbmc.LOGDEBUG)
    
    try:
        # Get connection pool
        pool = connection_pool_manager.get_pool()
        if pool is None:
            # Fallback to original implementation if pooling fails
            xbmc.log("[OPENAI] Connection pooling failed, using fallback", xbmc.LOGWARNING)
            return _call_fallback(prompt, model, api_key)
        
        start = time.time()
        # Use connection pool for request with improved timeout handling
        response = pool.request(
            'POST',
            url,
            body=json.dumps(data).encode('utf-8'),
            headers=headers,
            timeout=urllib3.Timeout(connect=3.0, read=20.0)  # Reduced timeouts with connection pooling
        )
        
        duration = time.time() - start
        
        if response.status >= 400:
            # Handle HTTP errors
            body = response.data.decode('utf-8', errors='ignore')
            xbmc.log(f"[OPENAI] HTTPError {response.status}: {body}", xbmc.LOGERROR)
            # Create exception similar to urllib.error.HTTPError for compatibility
            from urllib.error import HTTPError
            e = HTTPError(url, response.status, body, response.headers, None)
            e._http_status = response.status
            
            # Add specific error handling for common OpenAI API errors
            if response.status == 429:
                xbmc.log("[OPENAI] Rate limit exceeded. Please wait before sending another request.", xbmc.LOGERROR)
            elif response.status == 401:
                xbmc.log("[OPENAI] Authentication failed. Please check your API key.", xbmc.LOGERROR)
            elif response.status == 400:
                xbmc.log("[OPENAI] Bad request. Please check your request parameters.", xbmc.LOGERROR)
            elif response.status >= 500:
                xbmc.log(f"[OPENAI] Server error ({response.status}). Please try again later.", xbmc.LOGERROR)
                
            raise e
        
        payload = response.data.decode('utf-8')
        response_data = json.loads(payload)
        
        # OpenAI-compatible response shape - works with all OpenAI models including GPT-5
        content = response_data["choices"][0]["message"]["content"]
        xbmc.log(f"[OPENAI] Response in {duration:.2f}s: {repr(content[:60])}...", xbmc.LOGDEBUG)
        return content
        
    except Exception as e:
        xbmc.log(f"[OPENAI] Unexpected error: {type(e).__name__}: {str(e)}", xbmc.LOGERROR)
        xbmc.log(traceback.format_exc(), xbmc.LOGERROR)
        raise

def _call_fallback(prompt, model, api_key):
    """
    Fallback to original urllib.request implementation.
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
    
    # Log the request for debugging (using xbmc.log for consistency)
    xbmc.log(f"[OPENAI] Fallback request to model={model}, prompt={repr(prompt[:60])}...", xbmc.LOGDEBUG)
    
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
            xbmc.log(f"[OPENAI] Fallback response in {duration:.2f}s: {repr(content[:60])}...", xbmc.LOGDEBUG)
            return content
            
    except urllib.error.HTTPError as e:
        code = e.code
        # Read body to include in logs for diagnostics
        try:
            body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            body = ""
            
        xbmc.log(f"[OPENAI] Fallback HTTPError {code}: {body}", xbmc.LOGERROR)
        
        # Add specific error handling for common OpenAI API errors
        if code == 429:
            xbmc.log("[OPENAI] Rate limit exceeded. Please wait before sending another request.", xbmc.LOGERROR)
        elif code == 401:
            xbmc.log("[OPENAI] Authentication failed. Please check your API key.", xbmc.LOGERROR)
        elif code == 400:
            xbmc.log("[OPENAI] Bad request. Please check your request parameters.", xbmc.LOGERROR)
        elif code >= 500:
            xbmc.log(f"[OPENAI] Server error ({code}). Please try again later.", xbmc.LOGERROR)
            
        # Attach a neutral attribute for upstream inspection
        e._http_status = code
        raise e
    except Exception as e:
        xbmc.log(f"[OPENAI] Fallback unexpected error: {type(e).__name__}: {str(e)}", xbmc.LOGERROR)
        xbmc.log(traceback.format_exc(), xbmc.LOGERROR)
        raise
