import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from itertools import chain
from collections import defaultdict
import statistics
import xbmc

from .srt import parse_srt, group_blocks, write_srt
from .prompt import build_prompt, extract_translations
from .backoff import get_provider_name_from_fn

# Import connection pool manager for monitoring
try:
    from .connection_pool import get_connection_pool_manager, cleanup_connection_pools
    CONNECTION_POOL_AVAILABLE = True
except ImportError:
    CONNECTION_POOL_AVAILABLE = False

# Import batch retry optimizer for intelligent failure recovery
try:
    from .batch_retry_optimizer import get_batch_retry_optimizer, optimize_batch_failure_recovery
    BATCH_RETRY_OPTIMIZER_AVAILABLE = True
except ImportError:
    BATCH_RETRY_OPTIMIZER_AVAILABLE = False

# Import performance monitoring
try:
    from .performance_monitor import get_performance_monitor, record_batch_metrics, record_resource_usage
    PERFORMANCE_MONITORING_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITORING_AVAILABLE = False

# Import resource monitoring
try:
    from .resource_monitor import get_resource_monitor, log_resource_usage
    RESOURCE_MONITORING_AVAILABLE = True
except ImportError:
    RESOURCE_MONITORING_AVAILABLE = False

# Provider-specific batch sizing configuration with aggressive batch sizes for better throughput
PROVIDER_BATCH_CONFIG = {
    "OpenAI": {
        "max_batch_size": 75,  # Increased from 20 (3.75x improvement)
        "max_content_length": 24000,  # Increased from 8000 (3x improvement)
        "max_parallel": 8,  # Increased from 5
        "max_concurrent_groups": 5,  # Increased from 3
        "min_batch_size": 10,  # Minimum batch size for efficiency
        "adaptive_scaling": True,  # Enable content-aware batching
        "complexity_threshold": 0.7,  # Content complexity factor for batch size adjustment
        # Batch retry optimization settings
        "retry_optimizer_enabled": True,  # Enable intelligent batch retry
        "circuit_breaker_threshold": 5,  # Failures before circuit breaker trips
        "circuit_breaker_timeout": 60,  # Seconds before attempting reset
        "max_retry_attempts": 4,  # Maximum retry attempts per batch
        "retry_backoff_base": 1.0,  # Base delay for retry backoff
        "retry_backoff_max": 30.0,  # Maximum delay for retry backoff
    },
    "Gemini": {
        "max_batch_size": 45,  # Increased from 15 (3x improvement)
        "max_content_length": 30000,  # Increased from 10000 (3x improvement)
        "max_parallel": 6,  # Increased from 3
        "max_concurrent_groups": 4,  # Increased from 2
        "min_batch_size": 8,  # Minimum batch size for efficiency
        "adaptive_scaling": True,  # Enable content-aware batching
        "complexity_threshold": 0.6,  # Content complexity factor for batch size adjustment
        # Batch retry optimization settings
        "retry_optimizer_enabled": True,  # Enable intelligent batch retry
        "circuit_breaker_threshold": 4,  # More sensitive for Gemini
        "circuit_breaker_timeout": 45,  # Shorter timeout for Gemini
        "max_retry_attempts": 3,  # Fewer attempts for Gemini
        "retry_backoff_base": 1.5,  # Longer base delay for Gemini
        "retry_backoff_max": 25.0,  # Shorter max delay for Gemini
    },
    "OpenRouter": {
        "max_batch_size": 35,  # Increased from 10 (3.5x improvement)
        "max_content_length": 18000,  # Increased from 6000 (3x improvement)
        "max_parallel": 5,  # Increased from 3
        "max_concurrent_groups": 3,  # Maintained at 3 for stability
        "min_batch_size": 5,  # Minimum batch size for efficiency
        "adaptive_scaling": True,  # Enable content-aware batching
        "complexity_threshold": 0.5,  # Content complexity factor for batch size adjustment
        # Batch retry optimization settings
        "retry_optimizer_enabled": True,  # Enable intelligent batch retry
        "circuit_breaker_threshold": 6,  # More tolerant for OpenRouter
        "circuit_breaker_timeout": 90,  # Longer timeout for OpenRouter
        "max_retry_attempts": 5,  # More attempts for OpenRouter
        "retry_backoff_base": 2.0,  # Longer base delay for OpenRouter
        "retry_backoff_max": 45.0,  # Longer max delay for OpenRouter
    },
    "default": {
        "max_batch_size": 30,  # Increased from 15 (2x improvement)
        "max_content_length": 15000,  # Increased from 8000 (1.875x improvement)
        "max_parallel": 4,  # Increased from 3
        "max_concurrent_groups": 3,  # Increased from 2
        "min_batch_size": 5,  # Minimum batch size for efficiency
        "adaptive_scaling": True,  # Enable content-aware batching
        "complexity_threshold": 0.6,  # Content complexity factor for batch size adjustment
        # Batch retry optimization settings
        "retry_optimizer_enabled": True,  # Enable intelligent batch retry
        "circuit_breaker_threshold": 5,  # Default failure threshold
        "circuit_breaker_timeout": 60,  # Default timeout
        "max_retry_attempts": 3,  # Default retry attempts
        "retry_backoff_base": 1.0,  # Default base delay
        "retry_backoff_max": 30.0,  # Default max delay
    }
}

def get_provider_batch_config(call_fn):
    """Get provider-specific batch configuration with backward compatibility"""
    provider_name = get_provider_name_from_fn(call_fn)
    config = PROVIDER_BATCH_CONFIG.get(provider_name, PROVIDER_BATCH_CONFIG["default"])

    # Ensure backward compatibility by providing defaults for new parameters
    default_config = {
        "max_batch_size": 20,  # Conservative default
        "max_content_length": 8000,
        "max_parallel": 3,
        "max_concurrent_groups": 2,
        "min_batch_size": 1,
        "adaptive_scaling": False,  # Disabled by default for compatibility
        "complexity_threshold": 0.6,
        # Batch retry optimization defaults
        "retry_optimizer_enabled": True,  # Enable by default if available
        "circuit_breaker_threshold": 5,
        "circuit_breaker_timeout": 60,
        "max_retry_attempts": 3,
        "retry_backoff_base": 1.0,
        "retry_backoff_max": 30.0
    }

    # Merge with defaults to ensure all required keys exist
    for key, default_value in default_config.items():
        if key not in config:
            config[key] = default_value
            xbmc.log(f"[CONFIG] Added missing config key '{key}' = {default_value} for {provider_name}", xbmc.LOGDEBUG)

    return config

# Performance monitoring for batch processing
class BatchPerformanceMonitor:
    """Monitor and analyze batch processing performance"""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.session_start = time.time()
        self.batch_stats = []

    def record_batch_start(self, batch_id, batch_size, content_length, avg_complexity):
        """Record the start of batch processing"""
        self.batch_stats.append({
            'batch_id': batch_id,
            'batch_size': batch_size,
            'content_length': content_length,
            'avg_complexity': avg_complexity,
            'start_time': time.time(),
            'status': 'processing'
        })

    def record_batch_end(self, batch_id, success=True, error=None, items_processed=0):
        """Record the end of batch processing"""
        for stat in self.batch_stats:
            if stat['batch_id'] == batch_id:
                stat['end_time'] = time.time()
                stat['duration'] = stat['end_time'] - stat['start_time']
                stat['success'] = success
                stat['error'] = str(error) if error else None
                stat['items_processed'] = items_processed
                stat['status'] = 'completed' if success else 'failed'

                # Record metrics
                self.metrics['batch_durations'].append(stat['duration'])
                self.metrics['batch_sizes'].append(stat['batch_size'])
                self.metrics['success_rate'].append(1 if success else 0)
                self.metrics['throughput'].append(items_processed / stat['duration'] if stat['duration'] > 0 else 0)
                break

    def get_performance_summary(self):
        """Get a summary of performance metrics"""
        if not self.batch_stats:
            return {}

        total_batches = len(self.batch_stats)
        successful_batches = sum(1 for s in self.batch_stats if s.get('success', False))
        total_duration = time.time() - self.session_start
        total_items = sum(s.get('items_processed', 0) for s in self.batch_stats)

        summary = {
            'total_batches': total_batches,
            'successful_batches': successful_batches,
            'success_rate': successful_batches / total_batches if total_batches > 0 else 0,
            'total_duration': total_duration,
            'total_items_processed': total_items,
            'overall_throughput': total_items / total_duration if total_duration > 0 else 0,
            'avg_batch_size': statistics.mean(self.metrics['batch_sizes']) if self.metrics['batch_sizes'] else 0,
            'avg_batch_duration': statistics.mean(self.metrics['batch_durations']) if self.metrics['batch_durations'] else 0,
            'avg_throughput': statistics.mean(self.metrics['throughput']) if self.metrics['throughput'] else 0
        }

        # Add complexity correlation if available
        if any(s.get('avg_complexity') is not None for s in self.batch_stats):
            complexities = [s['avg_complexity'] for s in self.batch_stats if s.get('avg_complexity') is not None]
            durations = [s['duration'] for s in self.batch_stats if s.get('avg_complexity') is not None]
            if complexities and durations:
                summary['complexity_duration_correlation'] = statistics.correlation(complexities, durations) if len(complexities) > 1 else 0

        return summary

    def log_performance_report(self):
        """Log a detailed performance report"""
        summary = self.get_performance_summary()

        xbmc.log("[PERFORMANCE] === Batch Processing Performance Report ===", xbmc.LOGINFO)
        xbmc.log(f"[PERFORMANCE] Total batches: {summary.get('total_batches', 0)}", xbmc.LOGINFO)
        xbmc.log(f"[PERFORMANCE] Success rate: {summary.get('success_rate', 0):.2%}", xbmc.LOGINFO)
        xbmc.log(f"[PERFORMANCE] Total items processed: {summary.get('total_items_processed', 0)}", xbmc.LOGINFO)
        xbmc.log(f"[PERFORMANCE] Overall throughput: {summary.get('overall_throughput', 0):.2f} items/sec", xbmc.LOGINFO)
        xbmc.log(f"[PERFORMANCE] Average batch size: {summary.get('avg_batch_size', 0):.1f}", xbmc.LOGINFO)
        xbmc.log(f"[PERFORMANCE] Average batch duration: {summary.get('avg_batch_duration', 0):.2f}s", xbmc.LOGINFO)

        if 'complexity_duration_correlation' in summary:
            xbmc.log(f"[PERFORMANCE] Complexity-duration correlation: {summary['complexity_duration_correlation']:.3f}", xbmc.LOGINFO)

        # Log individual batch details for debugging
        for stat in self.batch_stats[-5:]:  # Last 5 batches
            status = "SUCCESS" if stat.get('success') else "FAILED"
            xbmc.log(f"[PERFORMANCE] Batch {stat['batch_id']}: {status} "
                     f"({stat['batch_size']} items, {stat.get('duration', 0):.2f}s)", xbmc.LOGDEBUG)

# Global performance monitor instance
performance_monitor = BatchPerformanceMonitor()

def safe_calculate_content_complexity(text):
    """Safely calculate content complexity with fallback"""
    try:
        return calculate_content_complexity(text)
    except Exception as e:
        xbmc.log(f"[BATCH] Complexity calculation failed, using default: {str(e)}", xbmc.LOGWARNING)
        return 0.5  # Default medium complexity

def safe_calculate_dynamic_batch_size(batch, max_content_length, provider_config=None):
    """Safely calculate dynamic batch size with fallback to original logic"""
    try:
        return calculate_dynamic_batch_size(batch, max_content_length, provider_config)
    except Exception as e:
        xbmc.log(f"[BATCH] Dynamic batch size calculation failed, using conservative fallback: {str(e)}", xbmc.LOGWARNING)
        # Fallback to original conservative logic
        if not batch:
            return 1
        total_length = sum(len("\n".join(b["lines"])) for _, b in batch)
        if total_length < max_content_length // 2:
            return min(len(batch), max(1, int(max_content_length / (total_length / len(batch) if len(batch) > 0 else 1))))
        current_size = len(batch)
        while current_size > 0:
            estimated_length = (total_length / len(batch)) * current_size
            if estimated_length <= max_content_length:
                return current_size
            current_size -= 1
        return 1

def safe_create_dynamic_batches(enumerated_blocks, call_fn):
    """Safely create dynamic batches with fallback mechanisms"""
    try:
        return create_adaptive_batches_with_splitting(enumerated_blocks, call_fn)
    except Exception as e:
        xbmc.log(f"[BATCH] Advanced batching failed, falling back to simple batching: {str(e)}", xbmc.LOGWARNING)
        # Fallback to original simple batching logic
        try:
            provider_config = get_provider_batch_config(call_fn)
            max_batch_size = provider_config.get("max_batch_size", 20)  # Use conservative default
            max_content_length = provider_config.get("max_content_length", 8000)

            batches = []
            current_batch = []
            current_content_length = 0

            for i, block in enumerated_blocks:
                block_content_length = len("\n".join(block["lines"]))

                if (current_batch and
                    (len(current_batch) >= max_batch_size or
                     current_content_length + block_content_length > max_content_length)):
                    batches.append(current_batch)
                    current_batch = []
                    current_content_length = 0

                current_batch.append((i, block))
                current_content_length += block_content_length

            if current_batch:
                batches.append(current_batch)

            xbmc.log(f"[BATCH] Fallback batching created {len(batches)} simple batches", xbmc.LOGINFO)
            return batches

        except Exception as fallback_error:
            xbmc.log(f"[BATCH] Fallback batching also failed: {str(fallback_error)}", xbmc.LOGERROR)
            # Ultimate fallback: one item per batch
            return [[(i, block)] for i, block in enumerated_blocks]

# Update function references to use safe versions
calculate_content_complexity = safe_calculate_content_complexity
calculate_dynamic_batch_size = safe_calculate_dynamic_batch_size
create_dynamic_batches = safe_create_dynamic_batches

def calculate_content_complexity(text):
    """Calculate content complexity score (0.0 to 1.0)"""
    if not text.strip():
        return 0.0

    words = text.split()
    if not words:
        return 0.0

    # Factor 1: Average word length (normalized)
    avg_word_length = sum(len(word) for word in words) / len(words)
    length_factor = min(avg_word_length / 10.0, 1.0)  # Normalize to 0-1

    # Factor 2: Lexical diversity (unique words / total words)
    unique_words = len(set(word.lower() for word in words))
    diversity_factor = unique_words / len(words)

    # Factor 3: Special characters and numbers
    special_chars = sum(1 for char in text if not char.isalnum() and not char.isspace())
    special_factor = min(special_chars / len(text), 1.0)

    # Factor 4: Sentence complexity (number of clauses/punctuation)
    punctuation_marks = sum(1 for char in text if char in '.,;:!?()[]{}')
    punctuation_factor = min(punctuation_marks / max(len(words) / 5, 1), 1.0)

    # Weighted complexity score
    complexity = (
        length_factor * 0.3 +
        diversity_factor * 0.3 +
        special_factor * 0.2 +
        punctuation_factor * 0.2
    )

    return min(complexity, 1.0)

def calculate_dynamic_batch_size(batch, max_content_length, provider_config=None):
    """Calculate appropriate batch size based on content length and complexity"""
    if not batch:
        return 1

    # Get provider-specific configuration
    if provider_config is None:
        provider_config = PROVIDER_BATCH_CONFIG["default"]

    min_batch_size = provider_config.get("min_batch_size", 1)
    complexity_threshold = provider_config.get("complexity_threshold", 0.6)
    adaptive_scaling = provider_config.get("adaptive_scaling", True)

    # Calculate content metrics for the batch
    batch_texts = ["\n".join(b["lines"]) for _, b in batch]
    total_length = sum(len(text) for text in batch_texts)

    # Calculate average complexity across all items in batch
    complexities = [calculate_content_complexity(text) for text in batch_texts]
    avg_complexity = sum(complexities) / len(complexities) if complexities else 0.0

    # Calculate length variation (coefficient of variation)
    if len(batch_texts) > 1:
        lengths = [len(text) for text in batch_texts]
        mean_length = sum(lengths) / len(lengths)
        variance = sum((l - mean_length) ** 2 for l in lengths) / len(lengths)
        length_variation = (variance ** 0.5) / mean_length if mean_length > 0 else 0.0
    else:
        length_variation = 0.0

    # Base batch size calculation
    base_batch_size = len(batch)

    # Apply complexity adjustment
    if adaptive_scaling and avg_complexity > complexity_threshold:
        # Reduce batch size for complex content
        complexity_factor = 1.0 - (avg_complexity - complexity_threshold) / (1.0 - complexity_threshold)
        base_batch_size = max(min_batch_size, int(base_batch_size * complexity_factor))

    # Apply length variation adjustment
    if length_variation > 0.5:  # High variation
        base_batch_size = max(min_batch_size, int(base_batch_size * 0.8))

    # Content length constraint
    if total_length > max_content_length:
        # Estimate average content per item
        avg_content_per_item = total_length / len(batch)
        max_items_by_length = int(max_content_length / avg_content_per_item)
        base_batch_size = min(base_batch_size, max(max_items_by_length, min_batch_size))

    # Ensure we don't exceed the configured maximum
    max_batch_size = provider_config.get("max_batch_size", 50)
    final_batch_size = min(base_batch_size, max_batch_size)

    # Ensure minimum batch size
    final_batch_size = max(final_batch_size, min_batch_size)

    # Log batch size calculation details for monitoring
    xbmc.log(f"[BATCH] Dynamic batch size: {final_batch_size}/{len(batch)} "
             f"(complexity: {avg_complexity:.2f}, variation: {length_variation:.2f}, "
             f"total_length: {total_length}/{max_content_length})", xbmc.LOGDEBUG)

    return final_batch_size

def translate_batch(batch, lang, model, api_key, call_fn, max_retries=3, batch_id=None):
    """Translate a batch with retry logic, error handling, and performance monitoring"""
    indexed_texts = [(i, "\n".join(b["lines"])) for i, b in batch]
    prompt = build_prompt(indexed_texts, lang)

    # Calculate batch metrics for performance monitoring
    batch_texts = [text for _, text in indexed_texts]
    content_length = sum(len(text) for text in batch_texts)
    avg_complexity = sum(calculate_content_complexity(text) for text in batch_texts) / len(batch_texts)

    # Record batch start in performance monitor
    if batch_id and PERFORMANCE_MONITORING_AVAILABLE:
        performance_monitor.record_batch_start(batch_id, len(batch), content_length, avg_complexity)

    start_time = time.time()
    success = False
    error = None
    items_processed = 0

    try:
        # Try to translate with retries
        for attempt in range(max_retries):
            try:
                response = call_fn(prompt, model, api_key)
                translations = extract_translations(response)  # musi zwracać dict: i -> text

                missing = [i for i, _ in batch if i not in translations]
                if missing:
                    xbmc.log(f"[TRANSLATION] Missing translations for indices: {missing}", xbmc.LOGWARNING)
                    xbmc.log("=== Prompt ===\n" + prompt, xbmc.LOGDEBUG)
                    xbmc.log("=== Response ===\n" + response, xbmc.LOGDEBUG)

                    # If we have some translations, we can still return them
                    partial_results = [(i, translations[i]) for i, _ in batch if i in translations]
                    items_processed = len(partial_results)

                    # If this is the last attempt, return what we have
                    if attempt == max_retries - 1:
                        xbmc.log(f"[TRANSLATION] Giving up on missing translations after {max_retries} attempts", xbmc.LOGERROR)
                        success = len(partial_results) > 0
                        return partial_results

                    # Wait before retrying
                    time.sleep(2 ** attempt)
                    continue

                items_processed = len(translations)
                success = True
                return [(i, translations[i]) for i, _ in batch if i in translations]

            except Exception as e:
                xbmc.log(f"[TRANSLATION] Batch translation failed (attempt {attempt + 1}/{max_retries}): {str(e)}", xbmc.LOGERROR)
                xbmc.log(traceback.format_exc(), xbmc.LOGERROR)
                error = e

                # If this is the last attempt, re-raise the exception
                if attempt == max_retries - 1:
                    raise

                # Wait before retrying
                time.sleep(2 ** attempt)

    finally:
        # Record batch end in performance monitor
        if batch_id and PERFORMANCE_MONITORING_AVAILABLE:
            performance_monitor.record_batch_end(batch_id, success, error, items_processed)

            # Record comprehensive batch metrics
            duration = time.time() - start_time
            provider_name = get_provider_name_from_fn(call_fn)
            record_batch_metrics(len(batch), duration, success, items_processed, provider_name)

        # Log batch performance
        duration = time.time() - start_time
        throughput = items_processed / duration if duration > 0 else 0
        xbmc.log(f"[BATCH] Processed batch {batch_id or 'unknown'}: {items_processed}/{len(batch)} items "
                  f"in {duration:.2f}s ({throughput:.2f} items/sec)", xbmc.LOGDEBUG)

    # This should never be reached due to the re-raise above
    return []


from itertools import chain

def execute_batch_group(group, lang, model, api_key, call_fn, max_workers=None):
    """Execute a group of batches with optimized thread pool usage and intelligent fallback"""
    # Use provided max_workers or default to number of batches
    if max_workers is None:
        max_workers = min(len(group), 10)  # Cap at 10 workers to prevent resource exhaustion

    results = []
    failed_batches = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks with batch IDs for performance monitoring
        future_to_batch = {}
        for i, batch in enumerate(group):
            batch_id = f"group_{id(group)}_{i}"
            future = executor.submit(translate_batch, batch, lang, model, api_key, call_fn, batch_id=batch_id)
            future_to_batch[future] = (batch, batch_id)

        # Process completed tasks as they finish
        for future in as_completed(future_to_batch):
            batch, batch_id = future_to_batch[future]
            try:
                batch_result = future.result()
                results.extend(batch_result)
            except Exception as e:
                xbmc.log(f"[TRANSLATION] Batch {batch_id} failed: {str(e)}", xbmc.LOGERROR)
                xbmc.log(traceback.format_exc(), xbmc.LOGERROR)
                # Store failed batch for intelligent retry
                failed_batches.append((batch, e, batch_id))

    # Handle failed batches with intelligent retry optimization
    if failed_batches:
        # Get provider configuration to check if retry optimization is enabled
        provider_config = get_provider_batch_config(call_fn)
        retry_optimizer_enabled = provider_config.get("retry_optimizer_enabled", True) and BATCH_RETRY_OPTIMIZER_AVAILABLE

        if retry_optimizer_enabled:
            xbmc.log(f"[TRANSLATION] Using intelligent batch retry optimization for {len(failed_batches)} failed batches", xbmc.LOGINFO)

            # Create provider-specific retry optimizer with configuration
            from .batch_retry_optimizer import BatchRetryOptimizer
            retry_optimizer = BatchRetryOptimizer(
                max_batch_size=provider_config.get("max_batch_size", 50),
                min_batch_size=provider_config.get("min_batch_size", 1),
                circuit_breaker_threshold=provider_config.get("circuit_breaker_threshold", 5),
                circuit_breaker_timeout=provider_config.get("circuit_breaker_timeout", 60)
            )

            recovered_items = 0
            total_items = sum(len(batch) for batch, _, _ in failed_batches)

            # Create a wrapper function for batch translation
            def translate_batch_wrapper(batch_items):
                """Wrapper function for the retry optimizer"""
                try:
                    # Use provider-specific max retries for individual batch attempts
                    max_retries = provider_config.get("max_retry_attempts", 3)
                    return translate_batch(batch_items, lang, model, api_key, call_fn, max_retries=max_retries)
                except Exception as e:
                    raise e

            # Process each failed batch with intelligent retry
            for batch, error, batch_id in failed_batches:
                xbmc.log(f"[BATCH_RETRY] Optimizing recovery for batch {batch_id} ({len(batch)} items)", xbmc.LOGDEBUG)

                # Prepare failure context for intelligent analysis
                failure_context = {
                    'error': error,
                    'batch_id': batch_id,
                    'batch_size': len(batch)
                }

                # Use the batch retry optimizer
                try:
                    recovered_results, recovery_time = optimize_batch_failure_recovery(
                        batch, translate_batch_wrapper, failure_context
                    )

                    if recovered_results:
                        results.extend(recovered_results)
                        batch_recovered = len(recovered_results)
                        recovered_items += batch_recovered

                        recovery_rate = batch_recovered / len(batch) if batch else 0
                        xbmc.log(f"[BATCH_RETRY] Batch {batch_id}: recovered {batch_recovered}/{len(batch)} items "
                                f"({recovery_rate:.1%}) in {recovery_time:.2f}s", xbmc.LOGINFO)
                    else:
                        xbmc.log(f"[BATCH_RETRY] Batch {batch_id}: failed to recover any items", xbmc.LOGWARNING)

                except Exception as e:
                    xbmc.log(f"[BATCH_RETRY] Optimizer failed for batch {batch_id}: {str(e)}", xbmc.LOGERROR)
                    # Fall back to individual processing as last resort
                    xbmc.log(f"[FALLBACK] Processing {len(batch)} items individually for batch {batch_id}", xbmc.LOGDEBUG)

                    for item in batch:
                        try:
                            single_result = translate_batch([item], lang, model, api_key, call_fn,
                                                          max_retries=1, batch_id=f"{batch_id}_item_{item[0]}")
                            results.extend(single_result)
                            recovered_items += len(single_result)
                        except Exception as single_error:
                            xbmc.log(f"[FALLBACK] Failed to translate item {item[0]}: {str(single_error)}", xbmc.LOGERROR)

            overall_recovery_rate = recovered_items / total_items if total_items > 0 else 0
            xbmc.log(f"[BATCH_RETRY] Overall recovery: {recovered_items}/{total_items} items ({overall_recovery_rate:.1%})", xbmc.LOGINFO)

            # Log performance metrics
            retry_optimizer.log_performance_report()

        else:
            # Fallback to original logic if optimizer is not available
            xbmc.log(f"[TRANSLATION] Batch retry optimizer not available, using legacy fallback for {len(failed_batches)} failed batches", xbmc.LOGWARNING)
            recovered_items = 0
            total_items = sum(len(batch) for batch, _, _ in failed_batches)

            for batch, error, batch_id in failed_batches:
                batch_recovered = 0

                # Strategy 1: Try smaller batch sizes first
                if len(batch) > 1:
                    xbmc.log(f"[FALLBACK] Attempting smaller batch sizes for batch {batch_id}", xbmc.LOGDEBUG)

                    # Try half the batch size
                    half_size = max(1, len(batch) // 2)
                    for i in range(0, len(batch), half_size):
                        sub_batch = batch[i:i + half_size]
                        try:
                            sub_result = translate_batch(sub_batch, lang, model, api_key, call_fn,
                                                        max_retries=1, batch_id=f"{batch_id}_sub_{i//half_size}")
                            results.extend(sub_result)
                            batch_recovered += len(sub_result)
                        except Exception as e:
                            xbmc.log(f"[FALLBACK] Sub-batch failed: {str(e)}", xbmc.LOGDEBUG)
                            # Continue to individual processing

                # Strategy 2: Process each item individually
                remaining_items = len(batch) - batch_recovered
                if remaining_items > 0:
                    xbmc.log(f"[FALLBACK] Processing {remaining_items} items individually for batch {batch_id}", xbmc.LOGDEBUG)

                    for item_idx, item in enumerate(batch):
                        # Skip if already processed in sub-batch
                        if batch_recovered > 0 and item_idx < batch_recovered:
                            continue

                        try:
                            single_result = translate_batch([item], lang, model, api_key, call_fn,
                                                          max_retries=1, batch_id=f"{batch_id}_item_{item[0]}")
                            results.extend(single_result)
                            batch_recovered += len(single_result)
                        except Exception as e:
                            xbmc.log(f"[FALLBACK] Failed to translate item {item[0]}: {str(e)}", xbmc.LOGERROR)
                            # Continue processing other items

                recovered_items += batch_recovered
                recovery_rate = batch_recovered / len(batch) if batch else 0
                xbmc.log(f"[FALLBACK] Batch {batch_id}: recovered {batch_recovered}/{len(batch)} items ({recovery_rate:.1%})", xbmc.LOGINFO)

            overall_recovery_rate = recovered_items / total_items if total_items > 0 else 0
            xbmc.log(f"[FALLBACK] Overall recovery: {recovered_items}/{total_items} items ({overall_recovery_rate:.1%})", xbmc.LOGINFO)

    return results


def translate_in_batches(batches, lang, model, api_key, call_fn, parallel, report_progress=None, check_cancelled=None):
    """Translate batches with improved error handling, progress reporting, and performance monitoring"""
    results = []
    total = len(batches)
    done = 0

    # Get provider-specific configuration
    provider_config = get_provider_batch_config(call_fn)
    max_parallel = min(parallel, provider_config["max_parallel"])
    max_concurrent_groups = provider_config["max_concurrent_groups"]

    # Create all groups first
    batch_iter = iter(batches)
    groups = []
    group_count = 0

    def next_group():
        return list(islice(batch_iter, max_parallel))

    group = next_group()
    while group:
        group_count += 1
        groups.append((group_count, group))
        group = next_group()

    xbmc.log(f"[TRANSLATION] Created {len(groups)} groups for parallel processing", xbmc.LOGINFO)

    # Process groups in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_concurrent_groups) as executor:
        # Submit all groups for parallel processing
        future_to_group = {}
        for group_id, group in groups:
            if check_cancelled and check_cancelled():
                raise Exception("Translation interrupted by client")

            future = executor.submit(execute_batch_group, group, lang, model, api_key, call_fn)
            future_to_group[future] = (group_id, group)

        # Process completed groups as they finish
        for future in as_completed(future_to_group):
            group_id, group = future_to_group[future]

            if check_cancelled and check_cancelled():
                raise Exception("Translation interrupted by client")

            try:
                group_results = future.result()
                results.extend(group_results)

                # Update progress
                done += len(group)
                if report_progress:
                    report_progress(done, total)

                xbmc.log(f"[TRANSLATION] Completed group {group_id}: {len(group_results)} translations", xbmc.LOGINFO)

            except Exception as e:
                xbmc.log(f"[TRANSLATION] Group {group_id} failed: {str(e)}", xbmc.LOGERROR)
                xbmc.log(traceback.format_exc(), xbmc.LOGERROR)

                # Update progress even for failed groups
                done += len(group)
                if report_progress:
                    report_progress(done, total)

                xbmc.log(f"[TRANSLATION] Continuing despite failure in group {group_id}", xbmc.LOGWARNING)

    # Log performance summary after all batches are processed
    performance_monitor.log_performance_report()

    # Log connection pool statistics if available
    if CONNECTION_POOL_AVAILABLE:
        try:
            pool_manager = get_connection_pool_manager()
            pool_manager.log_all_stats()
        except Exception as e:
            xbmc.log(f"[CONNECTION_POOL] Failed to log connection pool stats: {str(e)}", xbmc.LOGWARNING)

    return results


def merge_translations(blocks, translated_pairs):
    """Merge translations with better error handling"""
    translated_map = dict(translated_pairs)
    merged = []
    
    success_count = 0
    failure_count = 0
    
    for i, block in enumerate(blocks):
        if i in translated_map:
            try:
                merged.append({**block, "lines": translated_map[i].split("\n")})
                success_count += 1
            except Exception as e:
                xbmc.log(f"[TRANSLATION] Failed to merge translation for block {i}: {str(e)}", xbmc.LOGERROR)
                # Keep original block if merge fails
                merged.append(block)
                failure_count += 1
        else:
            xbmc.log(f"[TRANSLATION] No translation found for block {i}", xbmc.LOGWARNING)
            # Keep original block if no translation
            merged.append(block)
            failure_count += 1
    
    xbmc.log(f"[TRANSLATION] Merge results: {success_count} successful, {failure_count} failed", xbmc.LOGINFO)
    return merged


def create_content_aware_batches(enumerated_blocks, call_fn):
    """Create batches dynamically based on content characteristics and provider capabilities"""
    provider_config = get_provider_batch_config(call_fn)
    max_batch_size = provider_config["max_batch_size"]
    max_content_length = provider_config["max_content_length"]
    min_batch_size = provider_config.get("min_batch_size", 1)
    adaptive_scaling = provider_config.get("adaptive_scaling", True)

    # Pre-analyze all blocks for content characteristics
    block_analysis = []
    for i, block in enumerated_blocks:
        text = "\n".join(block["lines"])
        content_length = len(text)
        complexity = calculate_content_complexity(text)
        block_analysis.append({
            'index': i,
            'block': block,
            'text': text,
            'length': content_length,
            'complexity': complexity,
            'efficiency_score': content_length / (complexity + 0.1)  # Higher = more efficient to batch
        })

    # Sort blocks by efficiency score for optimal batching
    if adaptive_scaling:
        block_analysis.sort(key=lambda x: x['efficiency_score'], reverse=True)

    batches = []
    current_batch = []
    current_content_length = 0
    current_complexity_sum = 0.0

    for analysis in block_analysis:
        i, block = analysis['index'], analysis['block']
        block_length = analysis['length']
        block_complexity = analysis['complexity']

        # Check if we should start a new batch
        should_start_new_batch = False

        if current_batch:
            # Test batch with new item
            test_batch = current_batch + [(i, block)]
            test_content_length = current_content_length + block_length
            test_complexity_avg = (current_complexity_sum + block_complexity) / len(test_batch)

            # Calculate dynamic batch size for test batch
            dynamic_size = calculate_dynamic_batch_size(test_batch, max_content_length, provider_config)

            # Start new batch if:
            # 1. Dynamic size calculation says we should
            # 2. Content length would exceed limit
            # 3. Batch is getting too complex
            if (len(test_batch) > dynamic_size or
                test_content_length > max_content_length or
                (adaptive_scaling and test_complexity_avg > provider_config.get("complexity_threshold", 0.7))):
                should_start_new_batch = True

        if should_start_new_batch:
            # Only create batch if it meets minimum size requirement
            if len(current_batch) >= min_batch_size:
                batches.append(current_batch)
            else:
                # Merge small batches if possible
                if batches and len(batches[-1]) + len(current_batch) <= max_batch_size:
                    batches[-1].extend(current_batch)
                else:
                    batches.append(current_batch)

            current_batch = []
            current_content_length = 0
            current_complexity_sum = 0.0

        # Add current item to batch
        current_batch.append((i, block))
        current_content_length += block_length
        current_complexity_sum += block_complexity

    # Handle the final batch
    if current_batch:
        if len(current_batch) >= min_batch_size:
            batches.append(current_batch)
        else:
            # Try to merge with previous batch
            if batches and len(batches[-1]) + len(current_batch) <= max_batch_size:
                batches[-1].extend(current_batch)
            else:
                batches.append(current_batch)

    # Sort batches back to original order for processing
    for batch in batches:
        batch.sort(key=lambda x: x[0])

    # Log batching statistics
    total_items = len(enumerated_blocks)
    avg_batch_size = total_items / len(batches) if batches else 0
    xbmc.log(f"[BATCH] Created {len(batches)} content-aware batches "
             f"(avg size: {avg_batch_size:.1f}, total items: {total_items})", xbmc.LOGINFO)

    return batches

def split_complex_subtitle(block, max_length_per_chunk=1000, max_complexity=0.8):
    """Split a complex subtitle block into smaller, more manageable chunks"""
    original_lines = block["lines"]
    original_text = "\n".join(original_lines)

    # If block is already small enough, return as-is
    if len(original_text) <= max_length_per_chunk and calculate_content_complexity(original_text) <= max_complexity:
        return [block]

    chunks = []
    current_chunk_lines = []
    current_chunk_length = 0

    for line in original_lines:
        line_length = len(line)

        # If adding this line would exceed chunk size, finalize current chunk
        if current_chunk_lines and current_chunk_length + line_length > max_length_per_chunk:
            if current_chunk_lines:
                chunk_text = "\n".join(current_chunk_lines)
                # Only create chunk if it's not too complex
                if calculate_content_complexity(chunk_text) <= max_complexity:
                    chunks.append({"lines": current_chunk_lines.copy()})
                else:
                    # Split further if still too complex
                    sub_chunks = split_complex_subtitle({"lines": current_chunk_lines}, max_length_per_chunk, max_complexity)
                    chunks.extend(sub_chunks)

            current_chunk_lines = []
            current_chunk_length = 0

        # If single line is too long, split it
        if line_length > max_length_per_chunk:
            # Split long line by sentences or phrases
            sentences = split_line_into_sentences(line)
            for sentence in sentences:
                if len(sentence) <= max_length_per_chunk:
                    chunks.append({"lines": [sentence]})
                else:
                    # Split very long sentences into smaller parts
                    words = sentence.split()
                    temp_line = ""
                    for word in words:
                        if len(temp_line + " " + word) > max_length_per_chunk and temp_line:
                            chunks.append({"lines": [temp_line.strip()]})
                            temp_line = word
                        else:
                            temp_line += " " + word if temp_line else word
                    if temp_line:
                        chunks.append({"lines": [temp_line.strip()]})
        else:
            current_chunk_lines.append(line)
            current_chunk_length += line_length

    # Handle remaining lines
    if current_chunk_lines:
        chunk_text = "\n".join(current_chunk_lines)
        if calculate_content_complexity(chunk_text) <= max_complexity:
            chunks.append({"lines": current_chunk_lines})
        else:
            # Final attempt to split complex remaining content
            sub_chunks = split_complex_subtitle({"lines": current_chunk_lines}, max_length_per_chunk, max_complexity)
            chunks.extend(sub_chunks)

    # Ensure we have at least one chunk
    if not chunks:
        chunks.append({"lines": original_lines[:1]})  # At least first line

    return chunks

def split_line_into_sentences(line):
    """Split a line into sentences or meaningful chunks"""
    # Simple sentence splitting - can be enhanced with better NLP
    sentences = []
    current_sentence = ""

    # Split by sentence endings
    words = line.split()
    for word in words:
        current_sentence += word + " "
        if word.endswith(('.', '!', '?', ':')) and len(current_sentence.strip()) > 20:
            sentences.append(current_sentence.strip())
            current_sentence = ""

    # Add remaining content
    if current_sentence.strip():
        sentences.append(current_sentence.strip())

    # If no sentences found, split by commas or just return original
    if not sentences:
        if ',' in line:
            sentences = [s.strip() for s in line.split(',') if s.strip()]
        else:
            sentences = [line]

    return sentences

def create_adaptive_batches_with_splitting(enumerated_blocks, call_fn):
    """Create batches with intelligent splitting for complex content"""
    provider_config = get_provider_batch_config(call_fn)
    max_content_length = provider_config["max_content_length"]

    # Split complex blocks first
    expanded_blocks = []
    for i, block in enumerated_blocks:
        block_text = "\n".join(block["lines"])

        # Check if block needs splitting
        if (len(block_text) > max_content_length * 0.8 or  # Very long
            calculate_content_complexity(block_text) > provider_config.get("complexity_threshold", 0.7)):  # Very complex

            xbmc.log(f"[BATCH] Splitting complex block {i} (length: {len(block_text)}, "
                     f"complexity: {calculate_content_complexity(block_text):.2f})", xbmc.LOGDEBUG)

            # Split the block
            chunks = split_complex_subtitle(block, max_content_length // 4, 0.6)
            for j, chunk in enumerate(chunks):
                expanded_blocks.append((f"{i}.{j}", chunk))
        else:
            expanded_blocks.append((i, block))

    xbmc.log(f"[BATCH] Expanded {len(enumerated_blocks)} blocks into {len(expanded_blocks)} chunks", xbmc.LOGINFO)

    # Now create content-aware batches from expanded blocks
    return create_content_aware_batches(expanded_blocks, call_fn)

# Update the main batch creation function to use intelligent splitting
def create_dynamic_batches(enumerated_blocks, call_fn):
    """Create dynamic batches with intelligent splitting for complex content"""
    return create_adaptive_batches_with_splitting(enumerated_blocks, call_fn)


def translate_subtitles(
    path,
    api_key,
    lang,
    model,
    call_fn,
    report_progress=None,
    check_cancelled=None,
    parallel=3
):
    """Translate subtitles with all optimizations"""
    xbmc.log(f"[TRANSLATION] Starting translation of {path}", xbmc.LOGINFO)
    start_time = time.time()

    # Record initial resource usage
    if RESOURCE_MONITORING_AVAILABLE:
        try:
            resource_monitor = get_resource_monitor()
            current_snapshot = resource_monitor.get_current_snapshot()
            if current_snapshot and current_snapshot.process_info:
                initial_memory = current_snapshot.process_info.get('memory_rss_mb', 0)
                initial_threads = current_snapshot.process_info.get('threads', 0)
                record_resource_usage(initial_memory, initial_threads, 0)
        except Exception as e:
            xbmc.log(f"[TRANSLATION] Failed to record initial resource usage: {str(e)}", xbmc.LOGDEBUG)

    blocks = parse_srt(path)
    xbmc.log(f"[TRANSLATION] Parsed {len(blocks)} blocks from SRT", xbmc.LOGINFO)
    
    # Create dynamic batches based on content and provider capabilities
    enumerated_blocks = list(enumerate(blocks))
    batches = create_dynamic_batches(enumerated_blocks, call_fn)
    
    xbmc.log(f"[TRANSLATION] Created {len(batches)} batches for translation", xbmc.LOGINFO)
    
    # Translate batches
    translated_pairs = translate_in_batches(
        batches, lang, model, api_key, call_fn, parallel,
        report_progress, check_cancelled
    )
    
    xbmc.log(f"[TRANSLATION] Completed translation of {len(translated_pairs)} blocks", xbmc.LOGINFO)
    
    # Merge translations
    merged = merge_translations(blocks, translated_pairs)
    
    # Calculate success rate
    success_rate = len(translated_pairs) / len(blocks) * 100 if blocks else 0
    xbmc.log(f"[TRANSLATION] Translation success rate: {success_rate:.2f}%", xbmc.LOGINFO)
    
    # Write result
    new_path = path.replace(".srt", f".{lang.lower()}.translated.srt")
    write_result = write_srt(merged, new_path)

    end_time = time.time()
    duration = end_time - start_time

    # Record final resource usage
    if RESOURCE_MONITORING_AVAILABLE:
        try:
            resource_monitor = get_resource_monitor()
            current_snapshot = resource_monitor.get_current_snapshot()
            if current_snapshot and current_snapshot.process_info:
                final_memory = current_snapshot.process_info.get('memory_rss_mb', 0)
                final_threads = current_snapshot.process_info.get('threads', 0)

                # Get connection count from connection pool if available
                connection_count = 0
                if CONNECTION_POOL_AVAILABLE:
                    try:
                        pool_manager = get_connection_pool_manager()
                        metrics = pool_manager.get_all_metrics()
                        connection_count = sum(
                            provider_metrics.get('connections_created', 0)
                            for provider_metrics in metrics.get('providers', {}).values()
                        )
                    except:
                        pass

                record_resource_usage(final_memory, final_threads, connection_count)

                # Record overall translation metrics
                if PERFORMANCE_MONITORING_AVAILABLE:
                    performance_monitor.record_gauge('translation_duration_seconds', duration)
                    performance_monitor.record_gauge('total_subtitles_processed', len(blocks))
                    performance_monitor.record_gauge('translation_success_rate', success_rate / 100.0)

        except Exception as e:
            xbmc.log(f"[TRANSLATION] Failed to record final resource usage: {str(e)}", xbmc.LOGDEBUG)

    xbmc.log(f"[TRANSLATION] Translation completed in {duration:.2f} seconds", xbmc.LOGINFO)

    # Report final statistics
    if report_progress:
        report_progress(len(batches), len(batches))  # Mark as complete

    return write_result or new_path
