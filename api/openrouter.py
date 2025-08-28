import json
import time
import urllib.request
import urllib.error

# Import connection pooling functionality
try:
    from core.connection_pool import make_openrouter_request, get_connection_pool_manager
    CONNECTION_POOLING_AVAILABLE = True
except ImportError:
    CONNECTION_POOLING_AVAILABLE = False

# Import performance monitoring
try:
    from core.performance_monitor import record_api_call, record_cost_metrics
    PERFORMANCE_MONITORING_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITORING_AVAILABLE = False

# Import cost tracking
try:
    from core.cost_analyzer import track_openrouter_cost
    COST_TRACKING_AVAILABLE = True
except ImportError:
    COST_TRACKING_AVAILABLE = False


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
    logger(f"[OPENROUTER] Request to model={model}, prompt={repr(prompt[:60])}...")

    start_time = time.time()
    tokens_used = 0
    success = False

    # Try connection pooling first if available
    if CONNECTION_POOLING_AVAILABLE:
        try:
            content = make_openrouter_request(prompt, model, api_key, timeout=10)
            duration = time.time() - start_time
            success = True
            logger(f"[OPENROUTER] Response in {duration:.2f}s: {repr(content[:60])}...")

            # Record performance metrics
            if PERFORMANCE_MONITORING_AVAILABLE:
                record_api_call('openrouter', duration, success, tokens_used)

            # Track cost
            if COST_TRACKING_AVAILABLE and tokens_used > 0:
                track_openrouter_cost(model, tokens_used, 0, success)

            return content
        except Exception as e:
            logger(f"[OPENROUTER] Connection pooling failed, falling back to urllib: {str(e)}")
            # Fall through to urllib fallback

    # Fallback to original urllib.request implementation
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

    try:
        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
        with urllib.request.urlopen(req, timeout=10) as res:
            duration = time.time() - start_time
            payload = res.read().decode("utf-8")
            response = json.loads(payload)

            # Extract token usage if available
            if 'usage' in response:
                tokens_used = response['usage'].get('total_tokens', 0)

            # OpenAI-compatible response shape
            content = response["choices"][0]["message"]["content"]
            success = True
            logger(f"[OPENROUTER] Response in {duration:.2f}s: {repr(content[:60])}...")

            # Record performance metrics
            if PERFORMANCE_MONITORING_AVAILABLE:
                record_api_call('openrouter', duration, success, tokens_used)

            return content

    except urllib.error.HTTPError as e:
        duration = time.time() - start_time
        code = e.code
        # Read body to include in logs for diagnostics
        try:
            body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            body = ""
        logger(f"[OPENROUTER] HTTPError {code}: {body}")

        # Record performance metrics for failed call
        if PERFORMANCE_MONITORING_AVAILABLE:
            record_api_call('openrouter', duration, False, tokens_used)

        # Attach a neutral attribute for upstream inspection (pattern used in gemini_api)
        e._http_status = code
        raise e
    except Exception as e:
        duration = time.time() - start_time
        logger(f"[OPENROUTER] Unexpected error: {type(e).__name__}: {str(e)}")

        # Record performance metrics for failed call
        if PERFORMANCE_MONITORING_AVAILABLE:
            record_api_call('openrouter', duration, False, tokens_used)

        raise