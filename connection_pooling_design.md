# Connection Pooling Implementation Design

## Overview

This document outlines the implementation of connection pooling for the Sub-AI Translator application to improve performance by reusing HTTP connections instead of creating new ones for each API request.

## Current Implementation Issues

The current implementation uses `urllib.request` for HTTP requests, which creates a new TCP connection for each request. This approach has several performance drawbacks:

1. **TCP Handshake Overhead**: Each new connection requires a TCP handshake, adding latency
2. **TLS Negotiation Overhead**: For HTTPS connections, TLS negotiation adds additional latency
3. **Resource Consumption**: Creating and destroying connections consumes system resources
4. **No Connection Reuse**: No mechanism to reuse existing connections for subsequent requests

## Proposed Solution

Implement connection pooling using the `urllib3` library, which provides:

1. **Connection Reuse**: Connections are kept alive and reused for subsequent requests
2. **Pool Management**: Multiple connection pools for different hosts
3. **Thread Safety**: Safe for concurrent access from multiple threads
4. **Timeout Management**: Configurable timeouts for different phases of requests
5. **Retry Logic**: Built-in retry mechanisms that can be customized

## Implementation Approach

### 1. Add urllib3 Dependency

First, we need to add `urllib3` as a dependency to the project. This will require updating the project's dependency management files.

### 2. Create Connection Pool Manager

Create a centralized connection pool manager that can be shared across all API modules:

```python
# core/connection_pool.py
import urllib3
import xbmc

class ConnectionPoolManager:
    def __init__(self):
        self.pools = {}
        self.default_pool = self._create_pool()

    def _create_pool(self, num_pools=10, maxsize=20, retries=False):
        """Create a connection pool with specified settings"""
        try:
            pool = urllib3.PoolManager(
                num_pools=num_pools,
                maxsize=maxsize,
                retries=retries,  # We handle retries ourselves
                timeout=urllib3.Timeout(connect=5.0, read=30.0),
                block=False
            )
            xbmc.log(f"[CONNECTION_POOL] Created pool with num_pools={num_pools}, maxsize={maxsize}", xbmc.LOGDEBUG)
            return pool
        except Exception as e:
            xbmc.log(f"[CONNECTION_POOL] Failed to create pool: {str(e)}", xbmc.LOGERROR)
            # Fallback to urllib.request if pooling fails
            return None

    def get_pool(self, host=None):
        """Get a connection pool for a specific host"""
        if self.default_pool is None:
            return None

        if host is None:
            return self.default_pool

        if host not in self.pools:
            try:
                pool = urllib3.ProxyManager(
                    proxy_url=host,
                    num_pools=4,
                    maxsize=10,
                    retries=False,
                    timeout=urllib3.Timeout(connect=5.0, read=30.0)
                ) if host.startswith('http') else self._create_pool()
                self.pools[host] = pool
                xbmc.log(f"[CONNECTION_POOL] Created pool for host: {host}", xbmc.LOGDEBUG)
            except Exception as e:
                xbmc.log(f"[CONNECTION_POOL] Failed to create pool for {host}: {str(e)}", xbmc.LOGERROR)
                return self.default_pool

        return self.pools[host]

# Global instance
connection_pool_manager = ConnectionPoolManager()
```

### 3. Update API Modules

Update each API module to use the connection pool instead of `urllib.request`:

#### OpenAI API Module Update

```python
# api/openai.py
import json
import time
import xbmc
from core.connection_pool import connection_pool_manager

def call(prompt, model, api_key):
    """
    Make a chat completion request to OpenAI API using connection pooling.
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
            # Fallback to original implementation
            return _call_fallback(prompt, model, api_key)

        start = time.time()
        # Use connection pool for request
        response = pool.request(
            'POST',
            url,
            body=json.dumps(data).encode('utf-8'),
            headers=headers,
            timeout=urllib3.Timeout(connect=5.0, read=30.0)
        )

        duration = time.time() - start

        if response.status >= 400:
            # Handle HTTP errors
            body = response.data.decode('utf-8', errors='ignore')
            xbmc.log(f"[OPENAI] HTTPError {response.status}: {body}", xbmc.LOGERROR)
            # Create exception similar to urllib.error.HTTPError
            from urllib.error import HTTPError
            raise HTTPError(url, response.status, body, response.headers, None)

        payload = response.data.decode('utf-8')
        response_data = json.loads(payload)

        content = response_data["choices"][0]["message"]["content"]
        xbmc.log(f"[OPENAI] Response in {duration:.2f}s: {repr(content[:60])}...", xbmc.LOGDEBUG)
        return content

    except Exception as e:
        xbmc.log(f"[OPENAI] Unexpected error: {type(e).__name__}: {str(e)}", xbmc.LOGERROR)
        import traceback
        xbmc.log(traceback.format_exc(), xbmc.LOGERROR)
        raise

def _call_fallback(prompt, model, api_key):
    """Fallback to original urllib.request implementation"""
    import urllib.request
    import urllib.error

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
        start = time.time()
        with urllib.request.urlopen(req, timeout=30) as res:
            duration = time.time() - start
            payload = res.read().decode("utf-8")
            response = json.loads(payload)
            content = response["choices"][0]["message"]["content"]
            xbmc.log(f"[OPENAI] Fallback response in {duration:.2f}s: {repr(content[:60])}...", xbmc.LOGDEBUG)
            return content
    except Exception as e:
        xbmc.log(f"[OPENAI] Fallback error: {type(e).__name__}: {str(e)}", xbmc.LOGERROR)
        raise
```

Similar updates would be made to the Gemini and OpenRouter API modules.

### 4. Integration with Backoff Mechanism

The connection pooling implementation should work seamlessly with the existing backoff mechanism. The backoff decorator will still handle rate limiting and retries, but the underlying HTTP requests will use connection pooling for better performance.

### 5. Configuration Options

Add configuration options to allow users to adjust connection pooling settings:

```python
# core/config.py
# Add connection pool configuration
CONNECTION_POOL_CONFIG = {
    "num_pools": 10,
    "maxsize": 20,
    "connect_timeout": 5.0,
    "read_timeout": 30.0
}
```

## Benefits

1. **Reduced Latency**: Eliminates TCP handshake and TLS negotiation overhead for subsequent requests
2. **Improved Throughput**: More efficient use of system resources
3. **Better Scalability**: Can handle more concurrent requests with the same resources
4. **Fallback Support**: Maintains compatibility with existing code through fallback mechanisms

## Testing Strategy

1. **Unit Tests**: Test connection pool creation and management
2. **Integration Tests**: Test API modules with connection pooling
3. **Performance Tests**: Compare performance with and without connection pooling
4. **Error Handling Tests**: Ensure fallback mechanisms work correctly

## Rollout Plan

1. **Phase 1**: Implement connection pooling in a separate module
2. **Phase 2**: Update one API module (OpenAI) to use connection pooling
3. **Phase 3**: Test and optimize the implementation
4. **Phase 4**: Update remaining API modules
5. **Phase 5**: Add configuration options and documentation

## Monitoring

Add logging to track:

- Connection pool creation and usage
- Performance improvements
- Error rates and fallback usage
- Resource utilization

This will help ensure the optimization is working as expected and identify any issues that need to be addressed.
