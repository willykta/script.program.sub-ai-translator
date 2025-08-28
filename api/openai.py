import json
import urllib.request
import urllib.error
import time
import sys
import traceback

# Import connection pooling functionality
try:
    from core.connection_pool import make_openai_request, get_connection_pool_manager
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
    from core.cost_analyzer import track_openai_cost
    COST_TRACKING_AVAILABLE = True
except ImportError:
    COST_TRACKING_AVAILABLE = False

def call(prompt, model, api_key):
    """
    Make a chat completion request to OpenAI API using connection pooling.

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
    # Log the request for debugging (using print for compatibility)
    print(f"[OPENAI] Request to model={model}, prompt={repr(prompt[:60])}...", file=sys.stderr)

    start_time = time.time()
    tokens_used = 0
    success = False
    error_type = None

    # Try connection pooling first if available
    if CONNECTION_POOLING_AVAILABLE:
        try:
            content = make_openai_request(prompt, model, api_key, timeout=30)
            duration = time.time() - start_time
            success = True
            print(f"[OPENAI] Response in {duration:.2f}s: {repr(content[:60])}...", file=sys.stderr)

            # Record performance metrics
            if PERFORMANCE_MONITORING_AVAILABLE:
                record_api_call('openai', duration, success, tokens_used)

            # Track cost
            if COST_TRACKING_AVAILABLE and tokens_used > 0:
                track_openai_cost(model, tokens_used, 0, success)  # We don't have output token count

            return content
        except Exception as e:
            print(f"[OPENAI] Connection pooling failed, falling back to urllib: {str(e)}", file=sys.stderr)
            error_type = type(e).__name__
            # Fall through to urllib fallback

    # Fallback to original urllib.request implementation
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
        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
        # Add timeout handling (30 seconds)
        with urllib.request.urlopen(req, timeout=30) as res:
            duration = time.time() - start_time
            payload = res.read().decode("utf-8")
            response = json.loads(payload)

            # Extract token usage if available
            if 'usage' in response:
                tokens_used = response['usage'].get('total_tokens', 0)

            # OpenAI-compatible response shape - works with all OpenAI models including GPT-5
            content = response["choices"][0]["message"]["content"]
            success = True
            print(f"[OPENAI] Response in {duration:.2f}s: {repr(content[:60])}...", file=sys.stderr)

            # Record performance metrics
            if PERFORMANCE_MONITORING_AVAILABLE:
                record_api_call('openai', duration, success, tokens_used)

            # Track cost
            if COST_TRACKING_AVAILABLE and tokens_used > 0:
                track_openai_cost(model, tokens_used, 0, success)

            return content

    except urllib.error.HTTPError as e:
        duration = time.time() - start_time
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

        # Record performance metrics for failed call
        if PERFORMANCE_MONITORING_AVAILABLE:
            record_api_call('openai', duration, False, tokens_used)

        # Attach a neutral attribute for upstream inspection
        e._http_status = code
        raise e
    except Exception as e:
        duration = time.time() - start_time
        print(f"[OPENAI] Unexpected error: {type(e).__name__}: {str(e)}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

        # Record performance metrics for failed call
        if PERFORMANCE_MONITORING_AVAILABLE:
            record_api_call('openai', duration, False, tokens_used)

        raise
