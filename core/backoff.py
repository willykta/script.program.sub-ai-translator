import time
from functools import wraps
from random import uniform
import xbmc

_last_request_time = 0

def rate_limited_backoff_on_429(min_interval=6.5, retries=5, base_delay=2.0, max_delay=20.0):
    return lambda fn: _wrap(fn, min_interval, retries, base_delay, max_delay)

def _wrap(fn, min_interval, retries, base_delay, max_delay):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        global _last_request_time

        now = time.time()
        delta = now - _last_request_time
        if delta < min_interval:
            wait = min_interval - delta
            xbmc.log(f"[RATE LIMIT] Waiting {wait:.2f}s before next call", xbmc.LOGDEBUG)
            time.sleep(wait)

        for attempt in range(retries):
            try:
                result = fn(*args, **kwargs)
                _last_request_time = time.time()
                return result
            except Exception as e:
                status = getattr(e, "status", None) or getattr(e, "status_code", None)
                xbmc.log(f"[BACKOFF] Attempt {attempt + 1}, exception: {type(e).__name__} {str(e)}", xbmc.LOGERROR)
                xbmc.log(f"[BACKOFF] Status: {status}", xbmc.LOGERROR)
                if status != 429 or attempt == retries - 1:
                    raise
                delay = min(base_delay * (2 ** attempt), max_delay) + uniform(1, 4)
                xbmc.log(f"[BACKOFF] Sleeping for {delay:.2f}s", xbmc.LOGERROR)
                time.sleep(delay)
    return wrapped
