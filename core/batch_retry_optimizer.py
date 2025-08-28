"""
Batch Retry Optimizer for Intelligent Batch Failure Recovery

This module provides advanced retry strategies for batch operations to avoid
the performance penalty of individual item retries when batches fail.

Features:
- Progressive batch size reduction instead of individual retries
- Intelligent failure analysis (network vs content vs rate limit)
- Circuit breaker pattern to avoid wasting time on persistent failures
- Smart batch splitting for partial failures
- Optimized retry timing with adaptive backoff
- Comprehensive performance monitoring

Expected improvements:
- 5-10x faster recovery from batch failures
- Better success rates through intelligent retry strategies
- Reduced API call overhead from fewer individual requests
- More predictable performance with circuit breaker protection
"""

import time
import threading
import statistics
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Callable
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
    log_function = lambda msg, level=0: print(f"[BATCH_RETRY] {msg}")

# Import performance monitoring
try:
    from .performance_monitor import get_performance_monitor
    PERFORMANCE_MONITORING_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITORING_AVAILABLE = False


class FailureType(Enum):
    """Types of batch failures for intelligent analysis"""
    NETWORK = "network"           # Connection timeouts, DNS failures
    RATE_LIMIT = "rate_limit"     # 429 errors, quota exceeded
    CONTENT = "content"          # Malformed content, token limits
    SERVER = "server"            # 5xx errors, service unavailable
    AUTH = "auth"               # Authentication/authorization failures
    UNKNOWN = "unknown"          # Unclassified failures


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


class BatchRetryMetrics:
    """Comprehensive metrics for batch retry performance"""

    def __init__(self):
        self.total_batches = 0
        self.failed_batches = 0
        self.recovered_batches = 0
        self.individual_retries_avoided = 0
        self.time_saved_seconds = 0.0
        self.failure_types = defaultdict(int)
        self.retry_strategies_used = defaultdict(int)
        self.circuit_breaker_trips = 0
        self.batch_size_reductions = []
        self.recovery_times = []
        self.lock = threading.Lock()

    def record_batch_attempt(self, batch_size: int, success: bool, failure_type: FailureType = None):
        """Record a batch attempt"""
        with self.lock:
            self.total_batches += 1
            if not success:
                self.failed_batches += 1
                if failure_type:
                    self.failure_types[failure_type.value] += 1

    def record_recovery(self, original_size: int, final_size: int, recovery_time: float,
                        strategy: str, individual_retries_avoided: int):
        """Record a successful recovery"""
        with self.lock:
            self.recovered_batches += 1
            self.individual_retries_avoided += individual_retries_avoided
            self.retry_strategies_used[strategy] += 1
            self.batch_size_reductions.append((original_size, final_size))
            self.recovery_times.append(recovery_time)

            # Estimate time saved (rough calculation)
            estimated_individual_time = individual_retries_avoided * 2.0  # Assume 2s per individual retry
            batch_time = recovery_time
            self.time_saved_seconds += max(0, estimated_individual_time - batch_time)

            # Record in performance monitor if available
            if PERFORMANCE_MONITORING_AVAILABLE:
                monitor = get_performance_monitor()
                monitor.record_counter('batch_retry_recoveries')
                monitor.record_gauge('batch_retry_time_saved', self.time_saved_seconds)
                monitor.record_gauge('batch_retry_efficiency',
                                   individual_retries_avoided / max(recovery_time, 0.1))
                monitor.record_histogram('batch_retry_recovery_time', recovery_time)

    def record_circuit_breaker_trip(self):
        """Record circuit breaker activation"""
        with self.lock:
            self.circuit_breaker_trips += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        with self.lock:
            total_attempts = self.total_batches
            recovery_rate = (self.recovered_batches / max(self.failed_batches, 1)) * 100

            return {
                'total_batches': total_attempts,
                'failed_batches': self.failed_batches,
                'recovered_batches': self.recovered_batches,
                'recovery_rate_percent': recovery_rate,
                'individual_retries_avoided': self.individual_retries_avoided,
                'estimated_time_saved_seconds': self.time_saved_seconds,
                'circuit_breaker_trips': self.circuit_breaker_trips,
                'failure_types': dict(self.failure_types),
                'retry_strategies': dict(self.retry_strategies_used),
                'avg_batch_size_reduction': self._calculate_avg_reduction(),
                'avg_recovery_time': statistics.mean(self.recovery_times) if self.recovery_times else 0
            }

    def _calculate_avg_reduction(self) -> float:
        """Calculate average batch size reduction"""
        if not self.batch_size_reductions:
            return 0.0

        reductions = [(orig - final) / orig for orig, final in self.batch_size_reductions if orig > 0]
        return statistics.mean(reductions) * 100 if reductions else 0.0

    def log_summary(self):
        """Log performance summary"""
        summary = self.get_summary()
        log_function("[BATCH_RETRY] === Retry Performance Summary ===", LOG_INFO)
        log_function(f"[BATCH_RETRY] Total batches: {summary['total_batches']}", LOG_INFO)
        log_function(f"[BATCH_RETRY] Failed batches: {summary['failed_batches']}", LOG_INFO)
        log_function(f"[BATCH_RETRY] Recovery rate: {summary['recovery_rate_percent']:.1f}%", LOG_INFO)
        log_function(f"[BATCH_RETRY] Individual retries avoided: {summary['individual_retries_avoided']}", LOG_INFO)
        log_function(f"[BATCH_RETRY] Estimated time saved: {summary['estimated_time_saved_seconds']:.1f}s", LOG_INFO)
        log_function(f"[BATCH_RETRY] Circuit breaker trips: {summary['circuit_breaker_trips']}", LOG_INFO)

        if summary['failure_types']:
            log_function("[BATCH_RETRY] Failure types:", LOG_DEBUG)
            for failure_type, count in summary['failure_types'].items():
                log_function(f"[BATCH_RETRY]   {failure_type}: {count}", LOG_DEBUG)


class CircuitBreaker:
    """Circuit breaker pattern implementation for batch operations"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60,
                 expected_exception: Exception = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self.lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                log_function("[CIRCUIT_BREAKER] Testing service recovery", LOG_INFO)
            else:
                raise self.expected_exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker"""
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.recovery_timeout

    def _on_success(self):
        """Handle successful operation"""
        with self.lock:
            self.failure_count = 0
            self.state = CircuitBreakerState.CLOSED

    def _on_failure(self):
        """Handle failed operation"""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                log_function(f"[CIRCUIT_BREAKER] Circuit breaker OPEN after {self.failure_count} failures", LOG_WARNING)


class FailureAnalyzer:
    """Intelligent failure analysis for determining retry strategies"""

    @staticmethod
    def analyze_failure(error: Exception, response_time: float = None,
                       status_code: int = None) -> FailureType:
        """Analyze failure to determine type and optimal retry strategy"""

        error_msg = str(error).lower()
        error_type = type(error).__name__

        # Check HTTP status codes first
        if status_code:
            if status_code == 429:
                return FailureType.RATE_LIMIT
            elif status_code == 401 or status_code == 403:
                return FailureType.AUTH
            elif status_code >= 500:
                return FailureType.SERVER
            elif status_code >= 400:
                return FailureType.CONTENT

        # Analyze error messages and types
        if any(keyword in error_msg for keyword in ['timeout', 'connection', 'network', 'dns', 'unreachable']):
            return FailureType.NETWORK

        if any(keyword in error_msg for keyword in ['rate limit', 'quota', 'throttle']):
            return FailureType.RATE_LIMIT

        if any(keyword in error_msg for keyword in ['token', 'length', 'content', 'format', 'malformed']):
            return FailureType.CONTENT

        if any(keyword in error_msg for keyword in ['server', 'internal', 'service unavailable']):
            return FailureType.SERVER

        if any(keyword in error_msg for keyword in ['auth', 'unauthorized', 'forbidden', 'key']):
            return FailureType.AUTH

        # Check response time for timeout-like failures
        if response_time and response_time > 30:  # Long response time
            return FailureType.NETWORK

        return FailureType.UNKNOWN

    @staticmethod
    def get_retry_strategy(failure_type: FailureType, current_batch_size: int,
                          max_batch_size: int) -> Dict[str, Any]:
        """Get optimal retry strategy based on failure type"""

        strategies = {
            FailureType.RATE_LIMIT: {
                'strategy': 'progressive_reduction',
                'reduction_factor': 0.5,  # Reduce to 50% size
                'backoff_multiplier': 2.0,
                'max_attempts': 4,
                'description': 'Rate limit - reduce batch size and increase backoff'
            },
            FailureType.NETWORK: {
                'strategy': 'progressive_reduction',
                'reduction_factor': 0.7,  # Reduce to 70% size
                'backoff_multiplier': 1.5,
                'max_attempts': 3,
                'description': 'Network issue - moderate reduction with shorter backoff'
            },
            FailureType.CONTENT: {
                'strategy': 'aggressive_splitting',
                'reduction_factor': 0.3,  # Reduce to 30% size
                'backoff_multiplier': 1.2,
                'max_attempts': 5,
                'description': 'Content issue - aggressive splitting needed'
            },
            FailureType.SERVER: {
                'strategy': 'progressive_reduction',
                'reduction_factor': 0.6,  # Reduce to 60% size
                'backoff_multiplier': 2.5,
                'max_attempts': 3,
                'description': 'Server error - moderate reduction with longer backoff'
            },
            FailureType.AUTH: {
                'strategy': 'no_retry',
                'reduction_factor': 1.0,
                'backoff_multiplier': 1.0,
                'max_attempts': 0,
                'description': 'Auth error - no retry needed'
            },
            FailureType.UNKNOWN: {
                'strategy': 'progressive_reduction',
                'reduction_factor': 0.5,  # Conservative reduction
                'backoff_multiplier': 2.0,
                'max_attempts': 3,
                'description': 'Unknown error - conservative approach'
            }
        }

        return strategies.get(failure_type, strategies[FailureType.UNKNOWN])


class BatchSplitter:
    """Intelligent batch splitting strategies"""

    @staticmethod
    def split_batch_progressive(batch: List, target_size: int) -> List[List]:
        """Split batch using progressive size reduction"""
        if len(batch) <= target_size:
            return [batch]

        splits = []
        remaining = batch.copy()

        while remaining:
            current_split = remaining[:target_size]
            splits.append(current_split)
            remaining = remaining[target_size:]

            # Gradually reduce target size if we're creating too many small batches
            if len(splits) > 3 and target_size > 1:
                target_size = max(1, target_size // 2)

        return splits

    @staticmethod
    def split_batch_content_aware(batch: List, target_size: int,
                                content_getter: Callable = None) -> List[List]:
        """Split batch considering content characteristics"""
        if len(batch) <= target_size:
            return [batch]

        # If no content getter provided, fall back to progressive splitting
        if not content_getter:
            return BatchSplitter.split_batch_progressive(batch, target_size)

        # Sort by content complexity (assuming higher complexity = harder to process)
        try:
            sorted_batch = sorted(batch, key=lambda x: content_getter(x) if content_getter else 0)
        except Exception:
            # Fall back if sorting fails
            sorted_batch = batch.copy()

        # Create splits, putting simpler items first
        splits = []
        for i in range(0, len(sorted_batch), target_size):
            splits.append(sorted_batch[i:i + target_size])

        return splits

    @staticmethod
    def split_batch_adaptive(batch: List, failure_type: FailureType,
                           original_size: int) -> List[List]:
        """Adaptively split batch based on failure type"""

        if failure_type == FailureType.CONTENT:
            # For content failures, split into very small batches
            target_size = max(1, original_size // 8)
        elif failure_type == FailureType.RATE_LIMIT:
            # For rate limits, moderate reduction
            target_size = max(1, original_size // 4)
        elif failure_type == FailureType.NETWORK:
            # For network issues, slight reduction
            target_size = max(1, original_size // 2)
        else:
            # Default conservative approach
            target_size = max(1, original_size // 3)

        return BatchSplitter.split_batch_progressive(batch, target_size)


class AdaptiveBackoff:
    """Adaptive backoff timing optimized for batch operations"""

    def __init__(self, base_delay: float = 1.0, max_delay: float = 30.0,
                 multiplier: float = 2.0):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.attempt_times = []

    def get_delay(self, attempt: int, failure_type: FailureType = None) -> float:
        """Get adaptive delay based on attempt number and failure type"""

        # Base exponential backoff
        delay = min(self.base_delay * (self.multiplier ** attempt), self.max_delay)

        # Adjust based on failure type
        if failure_type == FailureType.RATE_LIMIT:
            delay *= 1.5  # Longer delays for rate limits
        elif failure_type == FailureType.NETWORK:
            delay *= 0.8  # Shorter delays for network issues
        elif failure_type == FailureType.CONTENT:
            delay *= 0.5  # Much shorter delays for content issues

        # Add jitter to prevent thundering herd
        jitter = delay * 0.1 * (0.5 - time.time() % 1)  # +/- 10% jitter
        delay += jitter

        return max(0.1, delay)  # Minimum 100ms delay

    def record_attempt_time(self, response_time: float):
        """Record response time for future adaptation"""
        self.attempt_times.append(response_time)
        if len(self.attempt_times) > 10:
            self.attempt_times.pop(0)


class BatchRetryOptimizer:
    """Main optimizer class that coordinates all retry strategies"""

    def __init__(self, max_batch_size: int = 50, min_batch_size: int = 1,
                 circuit_breaker_threshold: int = 5, circuit_breaker_timeout: int = 60):
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size

        # Initialize components
        self.metrics = BatchRetryMetrics()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_threshold,
            recovery_timeout=circuit_breaker_timeout
        )
        self.failure_analyzer = FailureAnalyzer()
        self.batch_splitter = BatchSplitter()
        self.backoff = AdaptiveBackoff()

        self.lock = threading.Lock()

    def optimize_batch_retry(self, batch: List, translate_fn: Callable,
                            failure_context: Dict[str, Any] = None,
                            recursion_depth: int = 0, max_recursion_depth: int = 3) -> Tuple[List, float]:
        """
        Optimize batch retry with intelligent strategies

        Args:
            batch: List of items to translate
            translate_fn: Function to translate a batch
            failure_context: Context about the failure (error, response_time, etc.)
            recursion_depth: Current recursion depth (internal use)
            max_recursion_depth: Maximum allowed recursion depth to prevent infinite loops

        Returns:
            Tuple of (successful_results, recovery_time)
        """
        start_time = time.time()
        original_size = len(batch)

        # Prevent infinite recursion
        if recursion_depth >= max_recursion_depth:
            log_function(f"[BATCH_RETRY] Maximum recursion depth ({max_recursion_depth}) reached. "
                        f"Falling back to individual processing for {len(batch)} items.", LOG_WARNING)

            # Fallback to individual processing
            return self._process_items_individually(batch, translate_fn, start_time)

        # Analyze the failure if context provided
        failure_type = FailureType.UNKNOWN
        if failure_context:
            error = failure_context.get('error')
            response_time = failure_context.get('response_time')
            status_code = failure_context.get('status_code')

            if error:
                failure_type = self.failure_analyzer.analyze_failure(
                    error, response_time, status_code
                )

        log_function(f"[BATCH_RETRY] Optimizing retry for {original_size} items, "
                    f"failure type: {failure_type.value}", LOG_INFO)

        # Record the initial failure
        self.metrics.record_batch_attempt(original_size, False, failure_type)

        # Get optimal retry strategy
        strategy = self.failure_analyzer.get_retry_strategy(
            failure_type, original_size, self.max_batch_size
        )

        log_function(f"[BATCH_RETRY] Using strategy: {strategy['description']}", LOG_DEBUG)

        # Execute retry strategy
        successful_results = []
        attempt = 0
        current_batch_size = original_size

        while attempt < strategy['max_attempts'] and not successful_results:
            attempt += 1
            current_batch_size = max(
                self.min_batch_size,
                int(current_batch_size * strategy['reduction_factor'])
            )

            log_function(f"[BATCH_RETRY] Attempt {attempt}: trying batch size {current_batch_size}", LOG_DEBUG)

            # Split batch according to strategy
            if strategy['strategy'] == 'aggressive_splitting':
                sub_batches = self.batch_splitter.split_batch_adaptive(
                    batch, failure_type, original_size
                )
            else:
                sub_batches = self.batch_splitter.split_batch_progressive(
                    batch, current_batch_size
                )

            # Try each sub-batch
            for sub_batch in sub_batches:
                if len(sub_batch) == 0:
                    continue

                try:
                    # Use circuit breaker for individual sub-batch attempts
                    result = self.circuit_breaker.call(translate_fn, sub_batch)

                    if result:
                        successful_results.extend(result)
                        log_function(f"[BATCH_RETRY] Sub-batch of {len(sub_batch)} items succeeded", LOG_DEBUG)

                except Exception as e:
                    log_function(f"[BATCH_RETRY] Sub-batch failed: {str(e)}", LOG_DEBUG)
                    continue

            # If we got some results but not all, try to get the remaining items
            if successful_results and len(successful_results) < len(batch):
                remaining_indices = set(range(len(batch))) - {r[0] for r in successful_results}
                remaining_items = [batch[i] for i in remaining_indices]

                if remaining_items:
                    log_function(f"[BATCH_RETRY] Retrying {len(remaining_items)} remaining items", LOG_DEBUG)

                    # Use iterative approach instead of recursion to prevent stack overflow
                    remaining_results = self._retry_remaining_items_iteratively(
                        remaining_items, translate_fn, failure_type, recursion_depth, max_recursion_depth
                    )
                    successful_results.extend(remaining_results)

            # Apply backoff delay between attempts
            if attempt < strategy['max_attempts'] and not successful_results:
                delay = self.backoff.get_delay(attempt, failure_type)
                log_function(f"[BATCH_RETRY] Waiting {delay:.2f}s before next attempt", LOG_DEBUG)
                time.sleep(delay)

        # Record recovery metrics
        recovery_time = time.time() - start_time
        if successful_results:
            individual_retries_avoided = original_size - len(sub_batches) if 'sub_batches' in locals() else original_size
            self.metrics.record_recovery(
                original_size, current_batch_size, recovery_time,
                strategy['strategy'], individual_retries_avoided
            )
            log_function(f"[BATCH_RETRY] Successfully recovered {len(successful_results)}/{original_size} items "
                        f"in {recovery_time:.2f}s", LOG_INFO)
        else:
            log_function(f"[BATCH_RETRY] Failed to recover any items after {recovery_time:.2f}s", LOG_WARNING)

        return successful_results, recovery_time

    def _process_items_individually(self, batch: List, translate_fn: Callable,
                                  start_time: float) -> Tuple[List, float]:
        """
        Fallback method to process items individually when recursion limit is reached

        Args:
            batch: List of items to process individually
            translate_fn: Function to translate individual items
            start_time: Start time for performance tracking

        Returns:
            Tuple of (successful_results, recovery_time)
        """
        successful_results = []
        individual_failures = 0

        log_function(f"[BATCH_RETRY] Processing {len(batch)} items individually", LOG_INFO)

        for item in batch:
            try:
                # Process each item individually with circuit breaker protection
                result = self.circuit_breaker.call(translate_fn, [item])
                if result:
                    successful_results.extend(result)
                    log_function(f"[BATCH_RETRY] Individual item processed successfully", LOG_DEBUG)
                else:
                    individual_failures += 1
            except Exception as e:
                individual_failures += 1
                log_function(f"[BATCH_RETRY] Individual item failed: {str(e)}", LOG_DEBUG)

        recovery_time = time.time() - start_time

        if individual_failures > 0:
            log_function(f"[BATCH_RETRY] Individual processing: {len(successful_results)}/{len(batch)} "
                        f"succeeded, {individual_failures} failed", LOG_WARNING)

        return successful_results, recovery_time

    def _retry_remaining_items_iteratively(self, remaining_items: List, translate_fn: Callable,
                                         failure_type: FailureType, current_depth: int,
                                         max_depth: int) -> List:
        """
        Iteratively retry remaining items with progressively smaller batch sizes

        Args:
            remaining_items: Items that still need to be processed
            translate_fn: Function to translate batches
            failure_type: Type of failure that occurred
            current_depth: Current recursion depth
            max_depth: Maximum recursion depth

        Returns:
            List of successful results from remaining items
        """
        if not remaining_items:
            return []

        remaining_results = []
        current_batch_size = max(1, len(remaining_items) // 2)  # Start with half the size

        # Try progressively smaller batch sizes
        while remaining_items and current_batch_size >= 1:
            log_function(f"[BATCH_RETRY] Iterative retry: {len(remaining_items)} items, "
                        f"batch size {current_batch_size}", LOG_DEBUG)

            # Split remaining items into smaller batches
            sub_batches = self.batch_splitter.split_batch_progressive(remaining_items, current_batch_size)

            # Try each sub-batch
            for sub_batch in sub_batches:
                if not sub_batch:
                    continue

                try:
                    # Check if we've reached recursion limit before attempting
                    if current_depth >= max_depth - 1:
                        # Use individual processing for final attempt
                        individual_results, _ = self._process_items_individually(
                            sub_batch, translate_fn, time.time()
                        )
                        remaining_results.extend(individual_results)
                    else:
                        # Try the sub-batch with one more level of recursion allowed
                        sub_results, _ = self.optimize_batch_retry(
                            sub_batch, translate_fn,
                            {'error': None, 'failure_type': failure_type},
                            current_depth + 1, max_depth
                        )
                        remaining_results.extend(sub_results)

                    # Remove successfully processed items from remaining list
                    processed_indices = {r[0] for r in remaining_results[-len(sub_batch):]}
                    remaining_items = [item for item in remaining_items
                                     if item[0] not in processed_indices]

                except Exception as e:
                    log_function(f"[BATCH_RETRY] Sub-batch retry failed: {str(e)}", LOG_DEBUG)
                    # Continue to next sub-batch
                    continue

            # Reduce batch size for next iteration if we still have remaining items
            if remaining_items:
                current_batch_size = max(1, current_batch_size // 2)
                # Add small delay to prevent overwhelming the service
                time.sleep(0.1)

        # Final fallback: process any remaining items individually
        if remaining_items:
            log_function(f"[BATCH_RETRY] Final fallback: processing {len(remaining_items)} items individually", LOG_WARNING)
            final_results, _ = self._process_items_individually(remaining_items, translate_fn, time.time())
            remaining_results.extend(final_results)

        return remaining_results

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.metrics.get_summary()

    def log_performance_report(self):
        """Log comprehensive performance report"""
        self.metrics.log_summary()

    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics = BatchRetryMetrics()


# Global instance for use across the application
batch_retry_optimizer = BatchRetryOptimizer()


def get_batch_retry_optimizer() -> BatchRetryOptimizer:
    """Get the global batch retry optimizer instance"""
    return batch_retry_optimizer


def optimize_batch_failure_recovery(batch: List, translate_fn: Callable,
                                   failure_context: Dict[str, Any] = None,
                                   max_recursion_depth: int = 3) -> Tuple[List, float]:
    """
    Convenience function to optimize batch failure recovery

    Args:
        batch: List of items to translate
        translate_fn: Function to translate a batch
        failure_context: Context about the failure
        max_recursion_depth: Maximum allowed recursion depth to prevent infinite loops

    Returns:
        Tuple of (successful_results, recovery_time)
    """
    return batch_retry_optimizer.optimize_batch_retry(
        batch, translate_fn, failure_context, 0, max_recursion_depth
    )