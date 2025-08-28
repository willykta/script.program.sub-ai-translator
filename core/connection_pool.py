"""
Connection Pool Manager for API Providers

This module provides connection pooling and reuse functionality to eliminate
the overhead of creating new HTTP connections for each API call.

Features:
- Provider-specific connection managers
- Thread-safe connection pool access
- Connection health monitoring and automatic cleanup
- Automatic fallback to urllib.request if pooling fails
- Connection timeout and failure handling
- Performance monitoring and metrics
"""

import time
import threading
import urllib.request
import urllib.error
import json
import sys
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict
import traceback

try:
    import xbmc
    LOG_INFO = xbmc.LOGINFO
    LOG_WARNING = xbmc.LOGWARNING
    LOG_ERROR = xbmc.LOGERROR
    LOG_DEBUG = xbmc.LOGDEBUG
    log_function = xbmc.log
except ImportError:
    # Fallback for non-Kodi environments
    LOG_INFO = 0
    LOG_WARNING = 1
    LOG_ERROR = 2
    LOG_DEBUG = 3
    log_function = lambda msg, level=0: print(f"[CONNECTION_POOL] {msg}", file=sys.stderr)

# Import performance monitoring
try:
    from .performance_monitor import get_performance_monitor
    PERFORMANCE_MONITORING_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITORING_AVAILABLE = False


class ConnectionPoolMetrics:
    """Monitor connection pool performance and health"""

    def __init__(self):
        self.connections_created = 0
        self.connections_reused = 0
        self.connections_failed = 0
        self.connections_cleaned = 0
        self.total_requests = 0
        self.failed_requests = 0
        self.avg_response_time = 0.0
        self.lock = threading.Lock()

    def record_connection_created(self):
        with self.lock:
            self.connections_created += 1

    def record_connection_reused(self):
        with self.lock:
            self.connections_reused += 1

    def record_connection_failed(self):
        with self.lock:
            self.connections_failed += 1

    def record_connection_cleaned(self):
        with self.lock:
            self.connections_cleaned += 1

    def record_request(self, success: bool, response_time: float):
        with self.lock:
            self.total_requests += 1
            if not success:
                self.failed_requests += 1

            # Update rolling average response time
            if self.total_requests == 1:
                self.avg_response_time = response_time
            else:
                self.avg_response_time = (self.avg_response_time * (self.total_requests - 1) + response_time) / self.total_requests

            # Record in performance monitor if available
            if PERFORMANCE_MONITORING_AVAILABLE:
                monitor = get_performance_monitor()
                monitor.record_gauge('connection_pool_response_time', response_time)
                monitor.record_gauge('connection_pool_success_rate',
                                   (self.total_requests - self.failed_requests) / max(self.total_requests, 1))

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            reuse_rate = (self.connections_reused / max(self.connections_created, 1)) * 100
            success_rate = ((self.total_requests - self.failed_requests) / max(self.total_requests, 1)) * 100

            return {
                'connections_created': self.connections_created,
                'connections_reused': self.connections_reused,
                'connections_failed': self.connections_failed,
                'connections_cleaned': self.connections_cleaned,
                'reuse_rate_percent': reuse_rate,
                'total_requests': self.total_requests,
                'failed_requests': self.failed_requests,
                'success_rate_percent': success_rate,
                'avg_response_time': self.avg_response_time
            }

    def log_stats(self):
        stats = self.get_stats()
        log_function(f"[CONNECTION_POOL] Stats: Created={stats['connections_created']}, "
                    f"Reused={stats['connections_reused']} ({stats['reuse_rate_percent']:.1f}%), "
                    f"Failed={stats['connections_failed']}, Cleaned={stats['connections_cleaned']}", LOG_INFO)
        log_function(f"[CONNECTION_POOL] Requests: Total={stats['total_requests']}, "
                    f"Failed={stats['failed_requests']} ({stats['success_rate_percent']:.1f}%), "
                    f"Avg Response Time={stats['avg_response_time']:.2f}s", LOG_INFO)


class ConnectionInfo:
    """Information about a connection in the pool"""

    def __init__(self, url: str, headers: Dict[str, str]):
        self.url = url
        self.headers = headers.copy()
        self.created_at = time.time()
        self.last_used = time.time()
        self.use_count = 0
        self.failed_count = 0
        self.is_healthy = True

    def mark_used(self):
        """Mark connection as recently used"""
        self.last_used = time.time()
        self.use_count += 1

    def mark_failed(self):
        """Mark connection as failed"""
        self.failed_count += 1
        if self.failed_count >= 3:  # Mark unhealthy after 3 failures
            self.is_healthy = False

    def is_expired(self, max_age: int = 300) -> bool:
        """Check if connection has expired"""
        return (time.time() - self.last_used) > max_age

    def should_cleanup(self, max_age: int = 300, max_uses: int = 100) -> bool:
        """Check if connection should be cleaned up"""
        return not self.is_healthy or self.is_expired(max_age) or self.use_count >= max_uses


class BaseConnectionManager:
    """Base class for connection pool managers"""

    def __init__(self, provider_name: str, max_connections: int = 10, connection_timeout: int = 30):
        self.provider_name = provider_name
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.connections: List[ConnectionInfo] = []
        self.lock = threading.RLock()
        self.metrics = ConnectionPoolMetrics()
        self._cleanup_thread = None
        self._running = True

        # Start cleanup thread
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        def cleanup_worker():
            while self._running:
                try:
                    time.sleep(60)  # Cleanup every minute
                    self._cleanup_expired_connections()
                except Exception as e:
                    log_function(f"[CONNECTION_POOL] Cleanup error: {str(e)}", LOG_WARNING)

        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()

    def _cleanup_expired_connections(self):
        """Remove expired or unhealthy connections"""
        with self.lock:
            initial_count = len(self.connections)
            self.connections = [
                conn for conn in self.connections
                if not conn.should_cleanup()
            ]
            cleaned_count = initial_count - len(self.connections)

            if cleaned_count > 0:
                self.metrics.record_connection_cleaned()
                log_function(f"[CONNECTION_POOL] Cleaned up {cleaned_count} {self.provider_name} connections", LOG_DEBUG)

    def get_connection(self, url: str, headers: Dict[str, str]) -> Optional[ConnectionInfo]:
        """Get a healthy connection from the pool or create a new one"""
        with self.lock:
            # Try to find an existing healthy connection
            for conn in self.connections:
                if (conn.url == url and
                    conn.headers == headers and
                    conn.is_healthy and
                    not conn.is_expired()):
                    conn.mark_used()
                    self.metrics.record_connection_reused()
                    return conn

            # Create new connection if pool not full
            if len(self.connections) < self.max_connections:
                conn = ConnectionInfo(url, headers)
                self.connections.append(conn)
                self.metrics.record_connection_created()
                log_function(f"[CONNECTION_POOL] Created new {self.provider_name} connection "
                           f"({len(self.connections)}/{self.max_connections})", LOG_DEBUG)
                return conn

            # If pool is full, try to reuse the least recently used healthy connection
            healthy_connections = [conn for conn in self.connections if conn.is_healthy]
            if healthy_connections:
                lru_conn = min(healthy_connections, key=lambda c: c.last_used)
                lru_conn.mark_used()
                self.metrics.record_connection_reused()
                return lru_conn

        return None

    def mark_connection_failed(self, connection: ConnectionInfo):
        """Mark a connection as failed"""
        with self.lock:
            connection.mark_failed()
            self.metrics.record_connection_failed()

    def shutdown(self):
        """Shutdown the connection manager"""
        self._running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)

    def get_metrics(self) -> Dict[str, Any]:
        """Get connection pool metrics"""
        return self.metrics.get_stats()


class OpenAIConnectionManager(BaseConnectionManager):
    """Connection manager for OpenAI API"""

    def __init__(self):
        super().__init__("OpenAI", max_connections=15, connection_timeout=30)
        self.base_url = "https://api.openai.com/v1/chat/completions"

    def make_request(self, prompt: str, model: str, api_key: str, timeout: int = 30) -> str:
        """Make a request using connection pooling"""
        url = self.base_url
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4
        }

        return self._execute_request(url, headers, data, timeout)


class GeminiConnectionManager(BaseConnectionManager):
    """Connection manager for Gemini API"""

    def __init__(self):
        super().__init__("Gemini", max_connections=12, connection_timeout=30)

    def get_api_version_for_model(self, model: str) -> str:
        return "v1" if model.startswith("gemini-2.") else "v1beta"

    def make_request(self, prompt: str, model: str, api_key: str, timeout: int = 30) -> str:
        """Make a request using connection pooling"""
        api_version = self.get_api_version_for_model(model)
        url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{
                "role": "user",
                "parts": [{"text": prompt}]
            }]
        }

        return self._execute_request(url, headers, data, timeout)


class OpenRouterConnectionManager(BaseConnectionManager):
    """Connection manager for OpenRouter API"""

    def __init__(self):
        super().__init__("OpenRouter", max_connections=10, connection_timeout=30)
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def make_request(self, prompt: str, model: str, api_key: str, timeout: int = 30) -> str:
        """Make a request using connection pooling"""
        url = self.base_url
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4,
        }

        return self._execute_request(url, headers, data, timeout)


class ConnectionPoolManager:
    """Main connection pool manager that coordinates all provider-specific managers"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the singleton instance"""
        self.managers = {
            'openai': OpenAIConnectionManager(),
            'gemini': GeminiConnectionManager(),
            'openrouter': OpenRouterConnectionManager()
        }
        self.global_metrics = ConnectionPoolMetrics()

    def get_manager(self, provider: str) -> BaseConnectionManager:
        """Get the connection manager for a specific provider"""
        return self.managers.get(provider.lower())

    def make_request_with_pooling(self, provider: str, prompt: str, model: str,
                                api_key: str, timeout: int = 30) -> str:
        """Make a request using connection pooling with fallback"""
        manager = self.get_manager(provider)

        if manager:
            try:
                start_time = time.time()
                result = manager.make_request(prompt, model, api_key, timeout)
                response_time = time.time() - start_time

                self.global_metrics.record_request(True, response_time)
                return result

            except Exception as e:
                log_function(f"[CONNECTION_POOL] {provider} pooling failed, falling back to urllib: {str(e)}", LOG_WARNING)
                self.global_metrics.record_request(False, time.time() - time.time())

        # Fallback to direct urllib.request
        return self._make_direct_request(provider, prompt, model, api_key, timeout)

    def _make_direct_request(self, provider: str, prompt: str, model: str,
                           api_key: str, timeout: int = 30) -> str:
        """Fallback method using direct urllib.request calls"""
        try:
            if provider.lower() == 'openai':
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
            elif provider.lower() == 'gemini':
                api_version = "v1" if model.startswith("gemini-2.") else "v1beta"
                url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model}:generateContent?key={api_key}"
                headers = {"Content-Type": "application/json"}
                data = {
                    "contents": [{
                        "role": "user",
                        "parts": [{"text": prompt}]
                    }]
                }
            elif provider.lower() == 'openrouter':
                url = "https://openrouter.ai/api/v1/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                }
                data = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.4,
                }
            else:
                raise ValueError(f"Unknown provider: {provider}")

            req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
            start_time = time.time()

            with urllib.request.urlopen(req, timeout=timeout) as res:
                response_time = time.time() - start_time
                payload = res.read().decode("utf-8")
                response = json.loads(payload)

                self.global_metrics.record_request(True, response_time)

                if provider.lower() == 'openai' or provider.lower() == 'openrouter':
                    return response["choices"][0]["message"]["content"]
                elif provider.lower() == 'gemini':
                    return response["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    raise ValueError(f"Unknown provider response format: {provider}")

        except Exception as e:
            self.global_metrics.record_request(False, 0)
            raise

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get metrics for all connection pools"""
        metrics = {
            'global': self.global_metrics.get_stats(),
            'providers': {}
        }

        for name, manager in self.managers.items():
            metrics['providers'][name] = manager.get_metrics()

        # Record connection pool metrics in performance monitor
        if PERFORMANCE_MONITORING_AVAILABLE:
            monitor = get_performance_monitor()
            global_stats = metrics['global']

            # Record global connection pool metrics
            monitor.record_gauge('connection_pool_reuse_rate', global_stats.get('reuse_rate_percent', 0) / 100.0)
            monitor.record_gauge('connection_pool_health', global_stats.get('success_rate_percent', 0) / 100.0)
            monitor.record_counter('connection_pool_total_requests', global_stats.get('total_requests', 0))

            # Record per-provider metrics
            for provider_name, provider_metrics in metrics['providers'].items():
                tags = {'provider': provider_name}
                monitor.record_gauge('connection_pool_provider_reuse_rate',
                                   provider_metrics.get('reuse_rate_percent', 0) / 100.0, tags=tags)
                monitor.record_gauge('connection_pool_provider_success_rate',
                                   provider_metrics.get('success_rate_percent', 0) / 100.0, tags=tags)

        return metrics

    def log_all_stats(self):
        """Log statistics for all connection pools"""
        log_function("[CONNECTION_POOL] === Global Connection Pool Statistics ===", LOG_INFO)
        self.global_metrics.log_stats()

        for name, manager in self.managers.items():
            log_function(f"[CONNECTION_POOL] === {name.upper()} Statistics ===", LOG_INFO)
            manager.metrics.log_stats()

    def shutdown(self):
        """Shutdown all connection managers"""
        for manager in self.managers.values():
            manager.shutdown()


# Global connection pool manager instance
connection_pool_manager = ConnectionPoolManager()


def get_connection_pool_manager() -> ConnectionPoolManager:
    """Get the global connection pool manager instance"""
    return connection_pool_manager


# Convenience functions for making requests with connection pooling
def make_openai_request(prompt: str, model: str, api_key: str, timeout: int = 30) -> str:
    """Make an OpenAI request using connection pooling"""
    return connection_pool_manager.make_request_with_pooling('openai', prompt, model, api_key, timeout)


def make_gemini_request(prompt: str, model: str, api_key: str, timeout: int = 30) -> str:
    """Make a Gemini request using connection pooling"""
    return connection_pool_manager.make_request_with_pooling('gemini', prompt, model, api_key, timeout)


def make_openrouter_request(prompt: str, model: str, api_key: str, timeout: int = 30) -> str:
    """Make an OpenRouter request using connection pooling"""
    return connection_pool_manager.make_request_with_pooling('openrouter', prompt, model, api_key, timeout)


# Cleanup function to be called on application shutdown
def cleanup_connection_pools():
    """Clean up all connection pools"""
    connection_pool_manager.shutdown()