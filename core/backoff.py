import time
from functools import wraps
from random import uniform
import xbmc

# Provider-specific rate limiting parameters
PROVIDER_CONFIG = {
    "OpenAI": {
        "min_interval": 0.02,  # Much more aggressive with connection pooling
        "retries": 2,          # Fewer retries needed with connection pooling
        "base_delay": 0.5,     # Reduced base delay
        "max_delay": 3.0,      # Reduced max delay
        "error_handlers": {
            429: {"strategy": "exponential", "multiplier": 1.5},  # More conservative rate limit handling
            502: {"strategy": "fixed", "delay": 1.0},             # Reduced delay
            503: {"strategy": "fixed", "delay": 2.0},             # Reduced delay
        }
    },
    "Gemini": {
        "min_interval": 0.05,  # More aggressive with connection pooling
        "retries": 2,          # Fewer retries needed
        "base_delay": 0.8,     # Reduced base delay
        "max_delay": 5.0,      # Reduced max delay
        "error_handlers": {
            429: {"strategy": "exponential", "multiplier": 2.0},  # Rate limit
            500: {"strategy": "fixed", "delay": 1.5},             # Reduced delay
            503: {"strategy": "fixed", "delay": 2.5},             # Reduced delay
        }
    },
    "OpenRouter": {
        "min_interval": 0.1,   # Much more aggressive with connection pooling
        "retries": 3,          # Slightly reduced retries
        "base_delay": 1.0,     # Reduced base delay
        "max_delay": 10.0,     # Reduced max delay
        "error_handlers": {
            429: {"strategy": "exponential", "multiplier": 2.0},  # Reduced multiplier
            502: {"strategy": "fixed", "delay": 1.5},             # Reduced delay
            503: {"strategy": "fixed", "delay": 3.0},             # Reduced delay
        }
    },
    "default": {
        "min_interval": 0.1,   # More aggressive default
        "retries": 2,          # Fewer retries
        "base_delay": 0.5,     # Reduced base delay
        "max_delay": 5.0,      # Reduced max delay
        "error_handlers": {
            429: {"strategy": "exponential", "multiplier": 1.5},  # More conservative rate limits
        }
    }
}

# Global tracking for rate limiting per provider
_last_request_time = {}

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

def _wrap(fn, min_interval, retries, base_delay, max_delay, provider=None):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        global _last_request_time
        
        # Determine provider for configuration
        provider_name = provider or get_provider_name(fn)
        config = PROVIDER_CONFIG.get(provider_name, PROVIDER_CONFIG["default"])
        
        # Use provided values or fall back to provider config
        effective_min_interval = min_interval if min_interval is not None else config["min_interval"]
        effective_retries = retries if retries is not None else config["retries"]
        effective_base_delay = base_delay if base_delay is not None else config["base_delay"]
        effective_max_delay = max_delay if max_delay is not None else config["max_delay"]
        
        # Rate limiting implementation with provider-specific tracking
        now = time.time()
        last_time = _last_request_time.get(provider_name, 0)
        delta = now - last_time
        
        if delta < effective_min_interval:
            wait = effective_min_interval - delta
            xbmc.log(f"[BACKOFF:{provider_name}] Rate limit waiting {wait:.2f}s before next call", xbmc.LOGINFO)
            time.sleep(wait)
        
        # Update last request time
        _last_request_time[provider_name] = time.time()
        
        # Retry loop with enhanced error handling
        for attempt in range(effective_retries):
            try:
                xbmc.log(f"[BACKOFF:{provider_name}] Attempt {attempt + 1}/{effective_retries}", xbmc.LOGDEBUG)
                result = fn(*args, **kwargs)
                if attempt > 0:
                    xbmc.log(f"[BACKOFF:{provider_name}] Success after {attempt + 1} attempts", xbmc.LOGINFO)
                return result
            except Exception as e:
                status = getattr(e, "_http_status", None) or getattr(e, "status", None) or getattr(e, "status_code", None)
                
                # Log error details
                xbmc.log(f"[BACKOFF:{provider_name}] Attempt {attempt + 1} failed with {type(e).__name__}: {str(e)}", xbmc.LOGERROR)
                if status:
                    xbmc.log(f"[BACKOFF:{provider_name}] HTTP Status: {status}", xbmc.LOGERROR)
                
                # Check if we should retry based on error type and provider config
                should_retry = False
                delay = 0
                
                if status and status in config["error_handlers"]:
                    handler = config["error_handlers"][status]
                    strategy = handler["strategy"]
                    
                    if strategy == "exponential":
                        delay = min(effective_base_delay * (handler["multiplier"] ** attempt), effective_max_delay)
                    elif strategy == "fixed":
                        delay = handler["delay"]
                    
                    # Add jitter to prevent thundering herd
                    delay += uniform(0.5, 2.0)
                    should_retry = True
                elif status == 429:
                    # Default rate limit handling if not specifically configured
                    delay = min(effective_base_delay * (2 ** attempt), effective_max_delay) + uniform(1, 3)
                    should_retry = True
                elif status and status >= 500:
                    # Generic server error handling
                    delay = min(effective_base_delay * (1.5 ** attempt), effective_max_delay) + uniform(0.5, 2.0)
                    should_retry = True
                
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
