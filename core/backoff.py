import time
from functools import wraps
from random import uniform
import xbmc

# Provider-specific rate limiting parameters - Optimized for performance
PROVIDER_CONFIG = {
    "OpenAI": {
        "min_interval": 0.03,  # Optimized: 30ms (33 RPS) - OpenAI supports 3,500+ RPM
        "retries": 3,
        "base_delay": 1.0,
        "max_delay": 8.0,
        "adaptive_enabled": True,  # Enable dynamic rate adjustment
        "timeout": 30,  # OpenAI recommends longer timeouts
        "connection_pool": True,  # Enable connection reuse
        "model_specific_rates": {  # Model-specific optimizations
            "gpt-4": 0.05,  # Slower for GPT-4 models
            "gpt-4o": 0.03,  # Faster for GPT-4o models
            "gpt-4o-mini": 0.02,  # Fastest for mini models
        },
        "error_handlers": {
            429: {"strategy": "adaptive_exponential", "multiplier": 2.0, "adaptive_factor": 1.5},  # Adaptive rate limit
            502: {"strategy": "fixed", "delay": 2.0},             # Bad gateway
            503: {"strategy": "fixed", "delay": 5.0},             # Service unavailable
            401: {"strategy": "no_retry", "delay": 0},            # Auth errors don't retry
        }
    },
    "Gemini": {
        "min_interval": 0.06,  # Optimized: 60ms (16 RPS) - Gemini supports 20+ RPM on higher tiers
        "retries": 3,
        "base_delay": 1.5,
        "max_delay": 10.0,
        "adaptive_enabled": True,  # Enable dynamic rate adjustment
        "timeout": 15,  # Gemini has moderate timeouts
        "connection_pool": False,  # Gemini API doesn't benefit from connection reuse
        "model_specific_rates": {  # Model-specific optimizations
            "gemini-2.0-flash": 0.04,  # Faster for Flash models
            "gemini-1.5-pro": 0.08,     # Slower for Pro models
            "gemini-1.5-flash": 0.06,   # Standard for Flash 1.5
        },
        "error_handlers": {
            429: {"strategy": "adaptive_exponential", "multiplier": 2.5, "adaptive_factor": 2.0},  # Adaptive rate limit
            500: {"strategy": "fixed", "delay": 3.0},             # Server error
            503: {"strategy": "fixed", "delay": 5.0},             # Service unavailable
            400: {"strategy": "no_retry", "delay": 0},            # Bad request doesn't retry
        }
    },
    "OpenRouter": {
        "min_interval": 0.25,  # Optimized: 250ms (4 RPS) - OpenRouter supports 20+ RPM on paid tiers
        "retries": 5,
        "base_delay": 2.0,
        "max_delay": 30.0,
        "adaptive_enabled": True,  # Enable dynamic rate adjustment
        "timeout": 20,  # OpenRouter has moderate timeouts
        "connection_pool": True,  # Enable connection reuse for better performance
        "model_specific_rates": {  # Provider-specific optimizations
            "openai/gpt-4o-mini": 0.2,  # OpenAI models through OpenRouter
            "anthropic/claude-3-haiku": 0.3,  # Anthropic models
            "meta-llama/llama-3.1-8b-instruct": 0.15,  # Llama models (often faster)
        },
        "error_handlers": {
            429: {"strategy": "adaptive_exponential", "multiplier": 3.0, "adaptive_factor": 2.5},  # Adaptive rate limit
            502: {"strategy": "fixed", "delay": 3.0},             # Bad gateway
            503: {"strategy": "fixed", "delay": 10.0},            # Service unavailable
            401: {"strategy": "no_retry", "delay": 0},            # Auth errors don't retry
        }
    },
    "default": {
        "min_interval": 0.1,  # More aggressive default
        "retries": 3,
        "base_delay": 1.0,
        "max_delay": 10.0,
        "adaptive_enabled": False,  # Disable for unknown providers
        "timeout": 10,  # Default timeout
        "connection_pool": False,  # Default connection handling
        "error_handlers": {
            429: {"strategy": "exponential", "multiplier": 2.0},  # Default for rate limits
        }
    }
}

# Global tracking for rate limiting per provider
_last_request_time = {}

# Dynamic rate adjustment tracking
_provider_stats = {}  # Track response times, errors, and success rates
_adaptive_intervals = {}  # Current adaptive intervals per provider
_rate_limit_hits = {}  # Track consecutive rate limit hits

def get_provider_name(fn):
    """Extract provider name from function for logging and configuration"""
    if hasattr(fn, '__name__'):
        name = fn.__name__
        if 'openai' in name.lower():
            return 'OpenAI'
        elif 'gemini' in name.lower():
            return 'Gemini'
        elif 'openrouter' in name.lower():
            return 'OpenRouter'
    return 'default'

def rate_limited_backoff_on_429(min_interval=None, retries=None, base_delay=None, max_delay=None, provider=None):
    """
    Enhanced rate limiting backoff decorator with provider-specific configurations.

    Args:
        min_interval: Minimum time between requests (overrides provider config if provided)
        retries: Number of retry attempts (overrides provider config if provided)
        base_delay: Base delay for backoff (overrides provider config if provided)
        max_delay: Maximum delay for backoff (overrides provider config if provided)
        provider: Provider name for specific configuration (optional)
    """
    def decorator(fn):
        return _wrap(fn, min_interval, retries, base_delay, max_delay, provider)
    return decorator

def rate_limited_backoff_on_429_with_model(min_interval=None, retries=None, base_delay=None, max_delay=None, provider=None, model_param='model'):
    """
    Enhanced rate limiting backoff decorator that extracts model information for optimization.

    Args:
        min_interval: Minimum time between requests (overrides provider config if provided)
        retries: Number of retry attempts (overrides provider config if provided)
        base_delay: Base delay for backoff (overrides provider config if provided)
        max_delay: Maximum delay for backoff (overrides provider config if provided)
        provider: Provider name for specific configuration (optional)
        model_param: Parameter name that contains the model name (default: 'model')
    """
    def decorator(fn):
        return _wrap_with_model(fn, min_interval, retries, base_delay, max_delay, provider, model_param)
    return decorator

def _wrap(fn, min_interval, retries, base_delay, max_delay, provider=None):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        return _wrap_with_model(fn, min_interval, retries, base_delay, max_delay, provider, None, *args, **kwargs)
    return wrapped

def _wrap_with_model(fn, min_interval, retries, base_delay, max_delay, provider, model_param, *args, **kwargs):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        global _last_request_time

        # Determine provider for configuration
        provider_name = provider or get_provider_name(fn)
        config = PROVIDER_CONFIG.get(provider_name, PROVIDER_CONFIG["default"])

        # Extract model information if available
        model = None
        if model_param and model_param in kwargs:
            model = kwargs[model_param]
        elif len(args) > 1 and hasattr(args[1], '__name__') and 'model' in str(args[1]).lower():
            model = args[1]  # Assume second argument is model

        # Use provided values or fall back to provider config
        effective_min_interval = min_interval if min_interval is not None else get_adaptive_interval(provider_name, model)
        effective_retries = retries if retries is not None else config["retries"]
        effective_base_delay = base_delay if base_delay is not None else config["base_delay"]
        effective_max_delay = max_delay if max_delay is not None else config["max_delay"]

        # Rate limiting implementation with provider-specific tracking
        now = time.time()
        last_time = _last_request_time.get(provider_name, 0)
        delta = now - last_time

        if delta < effective_min_interval:
            wait = effective_min_interval - delta
            xbmc.log(f"[BACKOFF:{provider_name}] Rate limit waiting {wait:.2f}s before next call (adaptive: {effective_min_interval:.3f}s)", xbmc.LOGDEBUG)
            time.sleep(wait)

        # Update last request time and track request start
        request_start_time = time.time()
        _last_request_time[provider_name] = request_start_time
        
        # Retry loop with enhanced error handling and response time tracking
        for attempt in range(effective_retries):
            try:
                xbmc.log(f"[BACKOFF:{provider_name}] Attempt {attempt + 1}/{effective_retries}", xbmc.LOGDEBUG)
                result = fn(*args, **kwargs)

                # Calculate response time
                response_time = time.time() - request_start_time

                # Update stats for successful request
                update_provider_stats(provider_name, response_time, success=True, rate_limit_hit=False)

                if attempt > 0:
                    xbmc.log(f"[BACKOFF:{provider_name}] Success after {attempt + 1} attempts", xbmc.LOGINFO)
                return result

            except Exception as e:
                # Calculate response time for failed request
                response_time = time.time() - request_start_time

                status = getattr(e, "_http_status", None) or getattr(e, "status", None) or getattr(e, "status_code", None)

                # Log error details
                xbmc.log(f"[BACKOFF:{provider_name}] Attempt {attempt + 1} failed with {type(e).__name__}: {str(e)}", xbmc.LOGERROR)
                if status:
                    xbmc.log(f"[BACKOFF:{provider_name}] HTTP Status: {status}", xbmc.LOGERROR)

                # Check if we should retry based on error type and provider config
                should_retry = False
                delay = 0
                is_rate_limit = status == 429

                if status and status in config["error_handlers"]:
                    handler = config["error_handlers"][status]
                    strategy = handler["strategy"]

                    if strategy == "no_retry":
                        # Don't retry for certain errors (auth, bad request, etc.)
                        should_retry = False
                        delay = 0
                        xbmc.log(f"[BACKOFF:{provider_name}] Not retrying {status} error as per provider config", xbmc.LOGINFO)
                    elif strategy == "adaptive_exponential":
                        # Adaptive backoff that increases only on rate limit hits
                        if is_rate_limit:
                            adaptive_factor = handler.get("adaptive_factor", 2.0)
                            delay = min(effective_base_delay * (handler["multiplier"] ** attempt) * adaptive_factor, effective_max_delay)
                        else:
                            # Use base delay for non-rate-limit errors
                            delay = min(effective_base_delay * (1.5 ** attempt), effective_max_delay)
                        should_retry = True
                    elif strategy == "exponential":
                        delay = min(effective_base_delay * (handler["multiplier"] ** attempt), effective_max_delay)
                        should_retry = True
                    elif strategy == "fixed":
                        delay = handler["delay"]
                        should_retry = True

                    # Add jitter to prevent thundering herd (only for strategies that retry)
                    if should_retry and strategy != "no_retry":
                        delay += uniform(0.5, 2.0)

                elif is_rate_limit:
                    # Adaptive rate limit handling
                    if config.get('adaptive_enabled', False):
                        # Increase adaptive interval on rate limit
                        current_interval = get_adaptive_interval(provider_name)
                        _adaptive_intervals[provider_name] = min(current_interval * 2, config["min_interval"] * 10)
                        delay = min(effective_base_delay * (2 ** attempt), effective_max_delay) + uniform(1, 3)
                    else:
                        # Default rate limit handling
                        delay = min(effective_base_delay * (2 ** attempt), effective_max_delay) + uniform(1, 3)
                    should_retry = True

                elif status and status >= 500:
                    # Generic server error handling
                    delay = min(effective_base_delay * (1.5 ** attempt), effective_max_delay) + uniform(0.5, 2.0)
                    should_retry = True

                # Update stats for failed request
                update_provider_stats(provider_name, response_time, success=False, rate_limit_hit=is_rate_limit)

                # If this is the last attempt or we shouldn't retry, raise the exception
                if not should_retry or attempt == effective_retries - 1:
                    xbmc.log(f"[BACKOFF:{provider_name}] Giving up after {attempt + 1} attempts", xbmc.LOGERROR)
                    raise

                # Log retry information
                xbmc.log(f"[BACKOFF:{provider_name}] Retrying in {delay:.2f}s", xbmc.LOGINFO)
                time.sleep(delay)
                
    return wrapped

# Connection reuse helper functions
# Note: urllib.request doesn't directly support connection reuse
# This is a placeholder for potential future implementation with other HTTP libraries
def get_connection(url):
    """
    Placeholder for connection reuse implementation.
    urllib.request doesn't directly support connection reuse.
    
    For future improvement, consider using the 'requests' library with Session objects
    which support connection pooling:
    
    session = requests.Session()
    session.mount('https://', HTTPAdapter(pool_connections=10, pool_maxsize=20))
    response = session.post(url, json=data, headers=headers)
    
    This would require adding 'requests' as a dependency to the project.
    """
    return None

def return_connection(url, conn):
    """
    Placeholder for connection reuse implementation.
    urllib.request doesn't directly support connection reuse.
    
    For future improvement, see get_connection() for details on using requests library
    with connection pooling.
    """
    pass

# Helper function to get provider name from function object
def get_provider_name_from_fn(fn):
    """Extract provider name from function object for configuration lookup"""
    return get_provider_name(fn)

# Test and validation functions
def validate_rate_limit_config():
    """Validate that rate limit configurations are reasonable"""
    issues = []

    for provider, config in PROVIDER_CONFIG.items():
        if provider == 'default':
            continue

        min_interval = config['min_interval']

        # Check for unreasonably aggressive settings
        if min_interval < 0.01:  # Less than 10ms
            issues.append(f"{provider}: min_interval {min_interval}s is too aggressive (< 10ms)")

        # Check for unreasonably conservative settings
        if min_interval > 2.0:  # More than 2 seconds
            issues.append(f"{provider}: min_interval {min_interval}s is too conservative (> 2s)")

        # Validate model-specific rates
        if 'model_specific_rates' in config:
            for model_pattern, rate in config['model_specific_rates'].items():
                if rate < 0.01 or rate > 1.0:
                    issues.append(f"{provider}: model {model_pattern} rate {rate}s is unreasonable")

    return issues

def test_rate_limiting_simulation(provider_name="OpenAI", num_requests=20, simulate_errors=False):
    """Simulate rate limiting behavior for testing"""
    import random

    print(f"Testing {provider_name} rate limiting with {num_requests} requests...")

    config = PROVIDER_CONFIG.get(provider_name, PROVIDER_CONFIG["default"])
    request_times = []
    wait_times = []

    start_time = time.time()

    for i in range(num_requests):
        now = time.time()
        last_time = _last_request_time.get(provider_name, 0)
        delta = now - last_time

        effective_interval = get_adaptive_interval(provider_name)

        if delta < effective_interval:
            wait = effective_interval - delta
            wait_times.append(wait)
            time.sleep(wait)  # In real test, this would be minimal
        else:
            wait_times.append(0)

        # Simulate API call
        call_start = time.time()
        time.sleep(0.05)  # Simulate 50ms API call

        # Simulate occasional errors if requested
        if simulate_errors and random.random() < 0.1:  # 10% error rate
            update_provider_stats(provider_name, time.time() - call_start, success=False, rate_limit_hit=random.random() < 0.3)
        else:
            update_provider_stats(provider_name, time.time() - call_start, success=True, rate_limit_hit=False)

        _last_request_time[provider_name] = time.time()
        request_times.append(time.time() - start_time)

    total_time = time.time() - start_time
    avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0

    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average wait time: {avg_wait:.3f}s")
    print(f"  Effective rate: {num_requests/total_time:.2f} RPS")
    print(f"  Final adaptive interval: {get_adaptive_interval(provider_name):.3f}s")

    return {
        'total_time': total_time,
        'avg_wait': avg_wait,
        'effective_rps': num_requests/total_time,
        'final_interval': get_adaptive_interval(provider_name)
    }

def update_provider_stats(provider_name, response_time, success=True, rate_limit_hit=False):
    """Update provider statistics for dynamic rate adjustment"""
    global _provider_stats, _adaptive_intervals, _rate_limit_hits

    if provider_name not in _provider_stats:
        _provider_stats[provider_name] = {
            'response_times': [],
            'success_count': 0,
            'error_count': 0,
            'rate_limit_count': 0,
            'last_update': time.time()
        }

    stats = _provider_stats[provider_name]
    config = PROVIDER_CONFIG.get(provider_name, PROVIDER_CONFIG["default"])

    # Update response time (keep last 10 measurements)
    stats['response_times'].append(response_time)
    if len(stats['response_times']) > 10:
        stats['response_times'].pop(0)

    # Update counters
    if success and not rate_limit_hit:
        stats['success_count'] += 1
    else:
        stats['error_count'] += 1
        if rate_limit_hit:
            stats['rate_limit_count'] += 1
            _rate_limit_hits[provider_name] = _rate_limit_hits.get(provider_name, 0) + 1
        else:
            # Reset rate limit hits on non-rate-limit errors
            _rate_limit_hits[provider_name] = 0

    stats['last_update'] = time.time()

    # Dynamic interval adjustment
    if config.get('adaptive_enabled', False):
        _adjust_adaptive_interval(provider_name)

def _adjust_adaptive_interval(provider_name):
    """Adjust the adaptive interval based on recent performance"""
    global _adaptive_intervals, _provider_stats, _rate_limit_hits

    stats = _provider_stats[provider_name]
    config = PROVIDER_CONFIG.get(provider_name, PROVIDER_CONFIG["default"])

    base_interval = config["min_interval"]
    current_adaptive = _adaptive_intervals.get(provider_name, base_interval)

    # Calculate average response time
    if stats['response_times']:
        avg_response_time = sum(stats['response_times']) / len(stats['response_times'])
    else:
        avg_response_time = 0

    # Get rate limit hit streak
    rate_limit_streak = _rate_limit_hits.get(provider_name, 0)

    # Adjust interval based on performance
    if rate_limit_streak > 0:
        # Increase interval on rate limit hits
        new_interval = min(current_adaptive * 1.5, base_interval * 5)
        xbmc.log(f"[ADAPTIVE:{provider_name}] Rate limit hit, increasing interval to {new_interval:.3f}s", xbmc.LOGINFO)
    elif stats['success_count'] > 5 and avg_response_time < 2.0:
        # Decrease interval on good performance
        new_interval = max(current_adaptive * 0.9, base_interval * 0.5)
        xbmc.log(f"[ADAPTIVE:{provider_name}] Good performance, decreasing interval to {new_interval:.3f}s", xbmc.LOGDEBUG)
    else:
        # Keep current interval
        new_interval = current_adaptive

    _adaptive_intervals[provider_name] = new_interval

def get_adaptive_interval(provider_name, model=None):
    """Get the current adaptive interval for a provider, considering model-specific rates"""
    config = PROVIDER_CONFIG.get(provider_name, PROVIDER_CONFIG["default"])

    # Get base interval (adaptive or static)
    if config.get('adaptive_enabled', False):
        base_interval = _adaptive_intervals.get(provider_name, config["min_interval"])
    else:
        base_interval = config["min_interval"]

    # Apply model-specific adjustments if available
    if model and 'model_specific_rates' in config:
        model_rates = config['model_specific_rates']
        for model_pattern, rate in model_rates.items():
            if model_pattern in model:
                # Adjust base interval by model-specific rate
                adjusted_interval = base_interval * (rate / config["min_interval"])
                xbmc.log(f"[MODEL-ADJUST:{provider_name}] Model {model} using interval {adjusted_interval:.3f}s (base: {base_interval:.3f}s)", xbmc.LOGDEBUG)
                return adjusted_interval

    return base_interval

def get_provider_timeout(provider_name):
    """Get the recommended timeout for a provider"""
    config = PROVIDER_CONFIG.get(provider_name, PROVIDER_CONFIG["default"])
    return config.get("timeout", 10)

def should_use_connection_pool(provider_name):
    """Check if connection pooling is recommended for a provider"""
    config = PROVIDER_CONFIG.get(provider_name, PROVIDER_CONFIG["default"])
    return config.get("connection_pool", False)

def get_rate_limit_metrics(provider_name=None):
    """Get current rate limiting metrics for monitoring"""
    if provider_name:
        # Get metrics for specific provider
        stats = _provider_stats.get(provider_name, {})
        config = PROVIDER_CONFIG.get(provider_name, PROVIDER_CONFIG["default"])

        total_requests = stats.get('success_count', 0) + stats.get('error_count', 0)
        success_rate = (stats.get('success_count', 0) / total_requests * 100) if total_requests > 0 else 0

        return {
            'provider': provider_name,
            'current_interval': get_adaptive_interval(provider_name),
            'base_interval': config["min_interval"],
            'success_rate': success_rate,
            'rate_limit_hits': stats.get('rate_limit_count', 0),
            'total_requests': total_requests,
            'avg_response_time': sum(stats.get('response_times', [])) / len(stats.get('response_times', [1])) if stats.get('response_times') else 0
        }
    else:
        # Get metrics for all providers
        all_metrics = {}
        for provider in PROVIDER_CONFIG.keys():
            if provider != 'default':
                all_metrics[provider] = get_rate_limit_metrics(provider)
        return all_metrics

def reset_provider_stats(provider_name=None):
    """Reset statistics for a provider or all providers"""
    global _provider_stats, _adaptive_intervals, _rate_limit_hits

    if provider_name:
        if provider_name in _provider_stats:
            _provider_stats[provider_name] = {
                'response_times': [],
                'success_count': 0,
                'error_count': 0,
                'rate_limit_count': 0,
                'last_update': time.time()
            }
        if provider_name in _adaptive_intervals:
            del _adaptive_intervals[provider_name]
        if provider_name in _rate_limit_hits:
            del _rate_limit_hits[provider_name]
    else:
        _provider_stats.clear()
        _adaptive_intervals.clear()
        _rate_limit_hits.clear()

def log_rate_limit_status():
    """Log current rate limiting status for all providers"""
    metrics = get_rate_limit_metrics()
    for provider, data in metrics.items():
        xbmc.log(f"[RATE-MONITOR:{provider}] Interval: {data['current_interval']:.3f}s, "
                f"Success: {data['success_rate']:.1f}%, "
                f"Rate-limits: {data['rate_limit_hits']}, "
                f"Requests: {data['total_requests']}, "
                f"Avg Response: {data['avg_response_time']:.2f}s", xbmc.LOGINFO)

def start_rate_limit_monitoring(interval_minutes=5):
    """Start periodic monitoring of rate limiting effectiveness"""
    import threading

    def monitor_loop():
        while True:
            try:
                log_rate_limit_status()
                # Check for providers that might need attention
                metrics = get_rate_limit_metrics()
                for provider, data in metrics.items():
                    if data['rate_limit_hits'] > 10 and data['success_rate'] < 80:
                        xbmc.log(f"[RATE-ALERT:{provider}] High rate limit hits ({data['rate_limit_hits']}) "
                                f"and low success rate ({data['success_rate']:.1f}%)", xbmc.LOGWARNING)
                    elif data['avg_response_time'] > 10.0:
                        xbmc.log(f"[RATE-ALERT:{provider}] High average response time "
                                f"({data['avg_response_time']:.2f}s)", xbmc.LOGWARNING)
            except Exception as e:
                xbmc.log(f"[RATE-MONITOR] Monitoring error: {str(e)}", xbmc.LOGERROR)

            time.sleep(interval_minutes * 60)

    # Start monitoring in background thread
    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()
    xbmc.log(f"[RATE-MONITOR] Started monitoring with {interval_minutes} minute intervals", xbmc.LOGINFO)

def get_rate_limit_health_report():
    """Generate a comprehensive health report for rate limiting"""
    report = {
        'timestamp': time.time(),
        'providers': {},
        'system_health': 'good'
    }

    metrics = get_rate_limit_metrics()
    for provider, data in metrics.items():
        health_status = 'good'

        # Determine health status based on metrics
        if data['rate_limit_hits'] > 20:
            health_status = 'critical'
        elif data['rate_limit_hits'] > 10 or data['success_rate'] < 85:
            health_status = 'warning'
        elif data['avg_response_time'] > 5.0:
            health_status = 'warning'

        if health_status != 'good' and report['system_health'] == 'good':
            report['system_health'] = health_status
        elif health_status == 'critical':
            report['system_health'] = 'critical'

        report['providers'][provider] = {
            **data,
            'health_status': health_status
        }

    return report

# Backward compatibility functions
def get_current_rate_limits():
    """Legacy function for backward compatibility"""
    return {provider: config["min_interval"] for provider, config in PROVIDER_CONFIG.items()}

def set_rate_limit(provider, interval):
    """Legacy function for backward compatibility - sets base interval"""
    if provider in PROVIDER_CONFIG:
        PROVIDER_CONFIG[provider]["min_interval"] = interval
        xbmc.log(f"[BACKOFF] Updated {provider} base interval to {interval}s", xbmc.LOGINFO)
