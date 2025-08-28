import json
import urllib.request
import urllib.error
import time

# Import connection pooling functionality
try:
    from core.connection_pool import make_gemini_request, get_connection_pool_manager
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
    from core.cost_analyzer import track_gemini_cost
    COST_TRACKING_AVAILABLE = True
except ImportError:
    COST_TRACKING_AVAILABLE = False

def get_api_version_for_model(model: str) -> str:
    return "v1" if model.startswith("gemini-2.") else "v1beta"

def call(prompt, model, api_key, logger=print):
    logger(f"[GEMINI] Request to model={model}, prompt={repr(prompt[:60])}...")

    start_time = time.time()
    tokens_used = 0
    success = False

    # Try connection pooling first if available
    if CONNECTION_POOLING_AVAILABLE:
        try:
            text = make_gemini_request(prompt, model, api_key, timeout=10)
            duration = time.time() - start_time
            success = True
            logger(f"[GEMINI] Response in {duration:.2f}s: {repr(text[:60])}...")

            # Record performance metrics
            if PERFORMANCE_MONITORING_AVAILABLE:
                record_api_call('gemini', duration, success, tokens_used)

            # Track cost
            if COST_TRACKING_AVAILABLE and tokens_used > 0:
                track_gemini_cost(model, tokens_used, 0, success)

            return text
        except Exception as e:
            logger(f"[GEMINI] Connection pooling failed, falling back to urllib: {str(e)}")
            # Fall through to urllib fallback

    # Fallback to original urllib.request implementation
    api_version = get_api_version_for_model(model)
    url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }]
    }

    try:
        req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=10) as res:
            duration = time.time() - start_time
            response = json.loads(res.read().decode())
            text = response["candidates"][0]["content"]["parts"][0]["text"]
            success = True
            logger(f"[GEMINI] Response in {duration:.2f}s: {repr(text[:60])}...")

            # Record performance metrics
            if PERFORMANCE_MONITORING_AVAILABLE:
                record_api_call('gemini', duration, success, tokens_used)

            # Track cost
            if COST_TRACKING_AVAILABLE and tokens_used > 0:
                track_gemini_cost(model, tokens_used, 0, success)

            return text
    except urllib.error.HTTPError as e:
        duration = time.time() - start_time
        code = e.code
        body = e.read().decode("utf-8", errors="ignore")
        logger(f"[GEMINI] HTTPError {code}: {body}")

        # Record performance metrics for failed call
        if PERFORMANCE_MONITORING_AVAILABLE:
            record_api_call('gemini', duration, False, tokens_used)

        e._http_status = code  # zamiast niedozwalanej .status
        raise e
    except Exception as e:
        duration = time.time() - start_time
        logger(f"[GEMINI] Unexpected error: {type(e).__name__}: {str(e)}")

        # Record performance metrics for failed call
        if PERFORMANCE_MONITORING_AVAILABLE:
            record_api_call('gemini', duration, False, tokens_used)

        raise
