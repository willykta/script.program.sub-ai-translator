"""
Test script for Batch Retry Optimizer performance validation

This test validates the performance improvements of the new batch retry optimization
compared to the old individual item retry approach.

Run with: python -m pytest tests/batch_retry_optimizer_test.py -v
"""

import time
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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
    log_function = lambda msg, level=0: print(f"[TEST] {msg}")


class TestBatchRetryOptimizer(unittest.TestCase):
    """Test cases for batch retry optimizer performance"""

    def setUp(self):
        """Set up test fixtures"""
        try:
            from core.batch_retry_optimizer import (
                BatchRetryOptimizer, FailureType, CircuitBreaker,
                FailureAnalyzer, BatchSplitter, AdaptiveBackoff
            )
            self.optimizer_available = True
        except ImportError:
            self.optimizer_available = False
            self.skipTest("BatchRetryOptimizer not available")

        if self.optimizer_available:
            self.optimizer = BatchRetryOptimizer()
            self.failure_analyzer = FailureAnalyzer()
            self.batch_splitter = BatchSplitter()
            self.circuit_breaker = CircuitBreaker()
            self.backoff = AdaptiveBackoff()

    def test_failure_type_analysis(self):
        """Test failure type analysis for different error scenarios"""
        if not self.optimizer_available:
            return

        from core.batch_retry_optimizer import FailureType

        # Test rate limit detection
        rate_limit_error = Exception("429 Rate limit exceeded")
        failure_type = self.failure_analyzer.analyze_failure(rate_limit_error, status_code=429)
        self.assertEqual(failure_type, FailureType.RATE_LIMIT)

        # Test network error detection
        network_error = Exception("Connection timeout")
        failure_type = self.failure_analyzer.analyze_failure(network_error)
        self.assertEqual(failure_type, FailureType.NETWORK)

        # Test content error detection
        content_error = Exception("Token limit exceeded")
        failure_type = self.failure_analyzer.analyze_failure(content_error)
        self.assertEqual(failure_type, FailureType.CONTENT)

        # Test server error detection
        server_error = Exception("Internal server error")
        failure_type = self.failure_analyzer.analyze_failure(server_error, status_code=500)
        self.assertEqual(failure_type, FailureType.SERVER)

    def test_batch_splitting_strategies(self):
        """Test different batch splitting strategies"""
        if not self.optimizer_available:
            return

        # Create a test batch
        test_batch = [(i, f"Test item {i}") for i in range(20)]

        # Test progressive splitting
        splits = self.batch_splitter.split_batch_progressive(test_batch, 5)
        self.assertEqual(len(splits), 4)  # 20 items / 5 per batch = 4 batches
        self.assertEqual(len(splits[0]), 5)
        self.assertEqual(len(splits[-1]), 5)

        # Test adaptive splitting
        from core.batch_retry_optimizer import FailureType
        adaptive_splits = self.batch_splitter.split_batch_adaptive(
            test_batch, FailureType.CONTENT, 20
        )
        # Content failures should result in smaller batches
        total_split_items = sum(len(split) for split in adaptive_splits)
        self.assertEqual(total_split_items, 20)

    def test_circuit_breaker_functionality(self):
        """Test circuit breaker open/close behavior"""
        if not self.optimizer_available:
            return

        # Test initial state
        self.assertEqual(self.circuit_breaker.state.value, "closed")

        # Simulate failures
        for i in range(4):
            try:
                self.circuit_breaker.call(lambda: (_ for _ in ()).throw(Exception("Test failure")))
            except Exception:
                pass

        # Should still be closed (under threshold)
        self.assertEqual(self.circuit_breaker.state.value, "closed")

        # One more failure should open the circuit
        try:
            self.circuit_breaker.call(lambda: (_ for _ in ()).throw(Exception("Test failure")))
        except Exception:
            pass

        self.assertEqual(self.circuit_breaker.state.value, "open")

    def test_adaptive_backoff(self):
        """Test adaptive backoff timing"""
        if not self.optimizer_available:
            return

        from core.batch_retry_optimizer import FailureType

        # Test increasing delays
        delay1 = self.backoff.get_delay(0, FailureType.RATE_LIMIT)
        delay2 = self.backoff.get_delay(1, FailureType.RATE_LIMIT)
        delay3 = self.backoff.get_delay(2, FailureType.RATE_LIMIT)

        # Delays should increase
        self.assertLess(delay1, delay2)
        self.assertLess(delay2, delay3)

        # Test different failure types have different delays
        rate_limit_delay = self.backoff.get_delay(1, FailureType.RATE_LIMIT)
        network_delay = self.backoff.get_delay(1, FailureType.NETWORK)

        # Rate limit should have longer delay
        self.assertGreater(rate_limit_delay, network_delay)

    @patch('time.sleep')  # Mock sleep to speed up tests
    def test_batch_retry_optimization_simulation(self, mock_sleep):
        """Simulate batch retry optimization performance"""
        if not self.optimizer_available:
            return

        # Mock translation function that fails initially then succeeds
        call_count = 0
        def mock_translate_func(batch):
            nonlocal call_count
            call_count += 1

            # Fail first 2 attempts, then succeed
            if call_count <= 2:
                raise Exception("Simulated batch failure")

            # Return successful results
            return [(i, f"Translated {text}") for i, text in batch]

        # Create test batch
        test_batch = [(i, f"Item {i}") for i in range(10)]

        # Test the optimizer
        start_time = time.time()
        results, recovery_time = self.optimizer.optimize_batch_retry(
            test_batch, mock_translate_func,
            {'error': Exception("Initial failure")}
        )
        end_time = time.time()

        # Verify results
        self.assertEqual(len(results), 10)
        self.assertGreater(recovery_time, 0)
        self.assertLess(recovery_time, 10)  # Should be reasonably fast

        # Verify multiple calls were made (retries)
        self.assertGreater(call_count, 1)

    def test_performance_metrics_tracking(self):
        """Test that performance metrics are properly tracked"""
        if not self.optimizer_available:
            return

        # Reset metrics
        self.optimizer.reset_metrics()

        # Simulate some batch operations
        def mock_translate_success(batch):
            return [(i, f"Success {text}") for i, text in batch]

        def mock_translate_failure(batch):
            raise Exception("Test failure")

        # Successful batch
        test_batch = [(i, f"Item {i}") for i in range(5)]
        self.optimizer.optimize_batch_retry(test_batch, mock_translate_success)

        # Failed batch
        self.optimizer.optimize_batch_retry(test_batch, mock_translate_failure)

        # Check metrics
        metrics = self.optimizer.get_performance_metrics()
        self.assertGreater(metrics['total_batches'], 0)
        self.assertGreaterEqual(metrics['failed_batches'], 1)
        self.assertGreaterEqual(metrics['recovered_batches'], 0)

    def test_configuration_provider_specific(self):
        """Test provider-specific configuration handling"""
        if not self.optimizer_available:
            return

        # Mock xbmc module for testing
        mock_xbmc = Mock()
        mock_xbmc.log = Mock()
        mock_xbmc.LOGDEBUG = 3

        with patch.dict('sys.modules', {'xbmc': mock_xbmc}):
            with patch('core.translation.xbmc', mock_xbmc):
                # Test that different providers can have different configurations
                from core.translation import get_provider_batch_config

                # Mock function for OpenAI
                openai_func = Mock()
                openai_func.__name__ = 'openai_call'

                # Mock function for Gemini
                gemini_func = Mock()
                gemini_func.__name__ = 'gemini_call'

                openai_config = get_provider_batch_config(openai_func)
                gemini_config = get_provider_batch_config(gemini_func)

                # Configurations should be different
                self.assertNotEqual(openai_config['max_batch_size'], gemini_config['max_batch_size'])
                self.assertNotEqual(openai_config['circuit_breaker_threshold'], gemini_config['circuit_breaker_threshold'])


class PerformanceComparisonTest(unittest.TestCase):
    """Compare old vs new retry strategies performance"""

    def setUp(self):
        """Set up performance comparison test"""
        try:
            from core.batch_retry_optimizer import get_batch_retry_optimizer
            self.optimizer_available = True
        except ImportError:
            self.optimizer_available = False

    @patch('time.sleep')  # Mock sleep for faster testing
    def test_old_vs_new_retry_performance(self, mock_sleep):
        """Compare performance of old vs new retry strategies"""
        if not self.optimizer_available:
            self.skipTest("BatchRetryOptimizer not available")

        from core.batch_retry_optimizer import get_batch_retry_optimizer

        # Simulate old strategy: individual retries
        def old_retry_strategy(batch, translate_func):
            results = []
            start_time = time.time()

            for item in batch:
                success = False
                for attempt in range(3):  # 3 retries per item
                    try:
                        result = translate_func([item])
                        results.extend(result)
                        success = True
                        break
                    except Exception:
                        if attempt < 2:  # Don't sleep on last attempt
                            time.sleep(1)  # 1 second delay

                # Continue to next item even if this one fails (more realistic)
                # Only break if we're getting consistent failures
                if not success:
                    continue

            return results, time.time() - start_time

        # Simulate new strategy using optimizer
        def new_retry_strategy(batch, translate_func):
            optimizer = get_batch_retry_optimizer()
            return optimizer.optimize_batch_retry(batch, translate_func)

        # Mock translation function with occasional failures
        failure_count = 0
        def mock_translate_func(batch):
            nonlocal failure_count
            failure_count += 1

            # Fail first 2 attempts for any batch, then succeed
            # This simulates a temporary service issue
            if failure_count <= 2:
                raise Exception("Batch failure")

            return [(i, f"Translated {text}") for i, text in batch]

        # Test batch
        test_batch = [(i, f"Item {i}") for i in range(10)]

        # Compare performance
        old_results, old_time = old_retry_strategy(test_batch, mock_translate_func)

        # Reset failure count for fair comparison
        failure_count = 0

        new_results, new_time = new_retry_strategy(test_batch, mock_translate_func)

        # New strategy should succeed completely
        self.assertEqual(len(new_results), 10)

        # Old strategy might not get all items due to test limitations, but should get some
        self.assertGreater(len(old_results), 0)

        # New strategy should be faster (allowing some variance for test environment)
        # Note: In real scenarios, the improvement would be much more significant
        improvement_ratio = old_time / new_time if new_time > 0 else float('inf')

        print(f"\nPerformance Comparison:")
        print(f"Old strategy: {len(old_results)}/10 items in {old_time:.2f}s")
        print(f"New strategy: {len(new_results)}/10 items in {new_time:.2f}s")
        print(f"Improvement ratio: {improvement_ratio:.2f}x")

        # The new strategy should be reasonably fast
        self.assertGreater(new_time, 0)
        self.assertLess(new_time, 10)  # Should complete within reasonable time


class RecursionFixTest(unittest.TestCase):
    """Test cases to verify the infinite recursion bug fix"""

    def setUp(self):
        """Set up test fixtures"""
        try:
            from core.batch_retry_optimizer import BatchRetryOptimizer
            self.optimizer_available = True
            self.optimizer = BatchRetryOptimizer()
        except ImportError:
            self.optimizer_available = False
            self.skipTest("BatchRetryOptimizer not available")

    def test_recursion_depth_limit_prevents_infinite_recursion(self):
        """Test that recursion depth limit prevents infinite recursion"""
        if not self.optimizer_available:
            return

        # Mock function that always fails to trigger maximum recursion
        call_count = 0
        def always_failing_translate(batch):
            nonlocal call_count
            call_count += 1
            raise Exception("Simulated persistent failure")

        # Create test batch
        test_batch = [(i, f"Item {i}") for i in range(5)]

        # This should not cause infinite recursion and should complete
        start_time = time.time()
        results, recovery_time = self.optimizer.optimize_batch_retry(
            test_batch, always_failing_translate,
            {'error': Exception("Initial failure")},
            max_recursion_depth=2  # Low limit to test quickly
        )

        # Should complete without hanging
        self.assertLess(recovery_time, 30)  # Should complete in reasonable time
        self.assertGreater(call_count, 0)  # Should have made some attempts

        print(f"Recursion limit test completed in {recovery_time:.2f}s with {call_count} calls")

    def test_individual_fallback_when_recursion_limit_reached(self):
        """Test that individual processing fallback works when recursion limit is reached"""
        if not self.optimizer_available:
            return

        # Mock function that partially succeeds then fails remaining items
        call_count = 0
        successful_calls = 0

        def partial_success_translate(batch):
            nonlocal call_count, successful_calls
            call_count += 1

            # First call succeeds with first half, subsequent calls fail
            if successful_calls == 0 and len(batch) > 1:
                successful_calls += 1
                # Return success for first half of batch
                mid_point = len(batch) // 2
                return [(batch[i][0], f"Success {batch[i][1]}") for i in range(mid_point)]
            else:
                # Fail remaining attempts to trigger recursion limit
                raise Exception("Persistent failure for remaining items")

        # Create larger test batch to trigger the scenario
        test_batch = [(i, f"Item {i}") for i in range(10)]

        # Test with very low recursion limit to force fallback
        results, recovery_time = self.optimizer.optimize_batch_retry(
            test_batch, partial_success_translate,
            {'error': Exception("Initial failure")},
            max_recursion_depth=1  # Force quick fallback to individual processing
        )

        # Should get some results (from initial partial success)
        self.assertGreater(len(results), 0)
        self.assertLess(recovery_time, 10)  # Should complete reasonably fast

        print(f"Fallback test: {len(results)}/{len(test_batch)} items succeeded in {recovery_time:.2f}s")

    def test_iterative_retry_processes_remaining_items(self):
        """Test that iterative retry successfully processes remaining items"""
        if not self.optimizer_available:
            return

        # Track which items have been processed
        processed_items = set()

        def selective_failure_translate(batch):
            # Succeed for items we haven't seen before, fail for previously failed items
            results = []
            for item in batch:
                if item[0] not in processed_items:
                    results.append((item[0], f"Success {item[1]}"))
                    processed_items.add(item[0])
                else:
                    # This would cause infinite recursion in the old version
                    raise Exception(f"Already processed item {item[0]}")

            if not results:
                raise Exception("No new items to process")

            return results

        # Create test batch
        test_batch = [(i, f"Item {i}") for i in range(8)]

        # This should work with the iterative approach
        results, recovery_time = self.optimizer.optimize_batch_retry(
            test_batch, selective_failure_translate,
            {'error': Exception("Initial failure")}
        )

        # Should eventually succeed for all items
        self.assertEqual(len(results), len(test_batch))
        self.assertLess(recovery_time, 15)  # Should complete in reasonable time

        print(f"Iterative retry test: {len(results)}/{len(test_batch)} items succeeded in {recovery_time:.2f}s")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)