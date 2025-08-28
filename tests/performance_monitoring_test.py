"""
Comprehensive tests for the performance monitoring and metrics system.

This test suite covers:
- Performance metrics collection and aggregation
- Resource utilization monitoring
- Cost tracking and analysis
- Real-time alerting system
- Performance dashboard and reporting
- Historical analysis and trends
- Export capabilities
- Anomaly detection
- Optimization recommendations
"""

import unittest
import time
import json
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
import sys

# Add the core module to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

# Mock xbmc for testing
sys.modules['xbmc'] = Mock()
sys.modules['xbmcaddon'] = Mock()
sys.modules['xbmcgui'] = Mock()

# Mock psutil for testing
psutil_mock = Mock()
psutil_mock.virtual_memory.return_value = Mock(percent=50.0, used=1024*1024*1024, available=1024*1024*1024)
psutil_mock.cpu_percent.return_value = 25.0
psutil_mock.cpu_count.return_value = 4
psutil_mock.cpu_count.logical = True
psutil_mock.cpu_count.physical = True
psutil_mock.Process.return_value.memory_info.return_value.rss = 100*1024*1024
psutil_mock.Process.return_value.cpu_percent.return_value = 10.0
psutil_mock.Process.return_value.num_threads.return_value = 5
psutil_mock.getloadavg.return_value = (1.0, 1.5, 2.0)
psutil_mock.cpu_freq.return_value = Mock(current=2500, min=800, max=3500)
psutil_mock.disk_usage.return_value = Mock(total=1000*1024**3, used=500*1024**3, free=500*1024**3, percent=50.0)
psutil_mock.net_io_counters.return_value = Mock(bytes_sent=1000, bytes_recv=2000, packets_sent=10, packets_recv=20)
psutil_mock.boot_time.return_value = time.time() - 3600
sys.modules['psutil'] = psutil_mock

class TestPerformanceMonitoring(unittest.TestCase):
    """Test cases for performance monitoring system"""

    def setUp(self):
        """Set up test fixtures"""
        # Reset global instances between tests
        from core.performance_monitor import performance_monitor
        from core.resource_monitor import resource_monitor
        from core.cost_analyzer import cost_tracker
        from core.performance_dashboard import performance_dashboard

        # Clear any existing data
        performance_monitor.reset_metrics()
        cost_tracker.usage_history.clear()
        cost_tracker.cost_history.clear()

    def test_performance_monitor_basic_functionality(self):
        """Test basic performance monitor functionality"""
        from core.performance_monitor import get_performance_monitor

        monitor = get_performance_monitor()

        # Test counter recording
        monitor.record_counter('test_counter', 5)
        self.assertEqual(monitor.counters['test_counter'], 5)

        # Test gauge recording
        monitor.record_gauge('test_gauge', 42.5)
        self.assertEqual(monitor.gauges['test_gauge'], 42.5)

        # Test histogram recording
        monitor.record_histogram('test_histogram', 10.0)
        self.assertIn(10.0, monitor.histograms['test_histogram'])

        # Test timer recording
        monitor.record_timer('test_timer', 2.5)
        self.assertIn(2.5, monitor.timers['test_timer'])

    def test_performance_monitor_alerts(self):
        """Test alert system functionality"""
        from core.performance_monitor import get_performance_monitor, AlertCondition, AlertSeverity

        monitor = get_performance_monitor()

        # Add an alert rule
        monitor.add_alert_rule(
            'test_gauge',
            AlertCondition.ABOVE_THRESHOLD,
            50.0,
            AlertSeverity.WARNING,
            "Test alert: {value} > {threshold}"
        )

        # Trigger alert
        monitor.record_gauge('test_gauge', 75.0)

        # Check that alert was created
        self.assertEqual(len(monitor.active_alerts), 1)
        alert_id = list(monitor.active_alerts.keys())[0]
        alert = monitor.active_alerts[alert_id]

        self.assertEqual(alert.metric_name, 'test_gauge')
        self.assertEqual(alert.current_value, 75.0)
        self.assertEqual(alert.threshold, 50.0)
        self.assertEqual(alert.severity, AlertSeverity.WARNING)

    def test_api_call_metrics_tracking(self):
        """Test API call metrics tracking"""
        from core.performance_monitor import record_api_call

        # Record successful API call
        record_api_call('openai', 1.5, True, 100)

        # Record failed API call
        record_api_call('openai', 2.0, False, 0)

        from core.performance_monitor import get_performance_monitor
        monitor = get_performance_monitor()

        # Check counters
        self.assertEqual(monitor.counters['api_calls'], 2)
        self.assertEqual(monitor.counters['api_errors'], 1)

        # Check timers
        self.assertIn(1.5, monitor.timers['api_call_duration'])
        self.assertIn(2.0, monitor.timers['api_call_duration'])

    def test_batch_metrics_tracking(self):
        """Test batch processing metrics tracking"""
        from core.performance_monitor import record_batch_metrics

        # Record successful batch
        record_batch_metrics(10, 5.0, True, 10, 'openai')

        # Record failed batch
        record_batch_metrics(8, 3.0, False, 0, 'openai')

        from core.performance_monitor import get_performance_monitor
        monitor = get_performance_monitor()

        # Check metrics
        self.assertEqual(monitor.counters['batches_processed'], 2)
        self.assertIn(10, monitor.histograms['batch_size'])
        self.assertIn(8, monitor.histograms['batch_size'])
        self.assertIn(5.0, monitor.timers['batch_processing_time'])
        self.assertIn(3.0, monitor.timers['batch_processing_time'])

    def test_resource_monitoring(self):
        """Test resource utilization monitoring"""
        from core.resource_monitor import get_resource_monitor, ResourceSnapshot

        monitor = get_resource_monitor()

        # Get current snapshot
        snapshot = monitor.get_current_snapshot()

        # Verify snapshot structure
        self.assertIsInstance(snapshot, ResourceSnapshot)
        self.assertIsInstance(snapshot.memory_info, dict)
        self.assertIsInstance(snapshot.cpu_info, dict)
        self.assertIsInstance(snapshot.process_info, dict)

        # Test resource summary
        summary = monitor.get_resource_summary(3600)
        self.assertIn('memory_stats', summary)
        self.assertIn('cpu_stats', summary)
        self.assertIn('process_stats', summary)

    def test_cost_tracking(self):
        """Test cost tracking functionality"""
        from core.cost_analyzer import get_cost_tracker

        tracker = get_cost_tracker()

        # Track API calls with costs
        tracker.track_api_call('openai', 'gpt-4', 1000, 500, True)
        tracker.track_api_call('gemini', 'gemini-pro', 800, 300, True)

        # Get cost summary
        summary = tracker.get_cost_summary(24)

        # Verify cost tracking
        self.assertGreater(summary['total_cost'], 0)
        self.assertEqual(summary['api_calls'], 2)
        self.assertIn('openai', summary['provider_breakdown'])
        self.assertIn('gemini', summary['provider_breakdown'])

    def test_cost_optimization_recommendations(self):
        """Test cost optimization recommendations"""
        from core.cost_analyzer import get_cost_tracker

        tracker = get_cost_tracker()

        # Add some cost data
        tracker.track_api_call('openai', 'gpt-4', 2000, 1000, True)
        tracker.track_api_call('gemini', 'gemini-pro', 500, 200, True)

        # Get recommendations
        recommendations = tracker.get_optimization_recommendations()

        # Should have recommendations
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)

    def test_performance_dashboard(self):
        """Test performance dashboard functionality"""
        from core.performance_dashboard import get_performance_dashboard

        dashboard = get_performance_dashboard()

        # Get system overview
        overview = dashboard.get_system_overview()

        # Verify overview structure
        self.assertIn('system_health', overview)
        self.assertIn('performance_score', overview)
        self.assertIn('cost_efficiency', overview)
        self.assertIn('active_alerts', overview)
        self.assertIn('recommendations', overview)

        # Test performance report generation
        report = dashboard.generate_performance_report(1)
        self.assertIn('generated_at', report)
        self.assertIn('system_overview', report)
        self.assertIn('performance_analysis', report)

    def test_benchmark_reporting(self):
        """Test benchmark report generation"""
        from core.performance_dashboard import get_performance_dashboard

        dashboard = get_performance_dashboard()

        # Create benchmark report
        benchmark = dashboard.create_benchmark_report(7, 1)  # 1 week baseline, 1 hour comparison

        # Verify benchmark structure
        self.assertIn('generated_at', benchmark)
        self.assertIn('baseline_period_hours', benchmark)
        self.assertIn('comparison_period_hours', benchmark)
        self.assertIn('benchmark_score', benchmark)

    def test_export_functionality(self):
        """Test data export functionality"""
        from core.performance_monitor import get_performance_monitor
        from core.cost_analyzer import get_cost_tracker
        from core.performance_dashboard import get_performance_dashboard

        # Test performance monitor export
        monitor = get_performance_monitor()
        monitor.record_counter('test_metric', 42)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            success = monitor.export_metrics(temp_file, 'json')
            self.assertTrue(success)

            # Verify exported data
            with open(temp_file, 'r') as f:
                data = json.load(f)
                self.assertIn('counters', data)
                self.assertEqual(data['counters']['test_metric'], 42)

        finally:
            os.unlink(temp_file)

        # Test dashboard export
        dashboard = get_performance_dashboard()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            temp_file = f.name

        try:
            success = dashboard.export_dashboard_report(temp_file, 'html')
            self.assertTrue(success)

            # Verify HTML export
            with open(temp_file, 'r') as f:
                content = f.read()
                self.assertIn('<!DOCTYPE html>', content)
                self.assertIn('Performance Dashboard', content)

        finally:
            os.unlink(temp_file)

    def test_anomaly_detection(self):
        """Test anomaly detection functionality"""
        from core.performance_monitor import get_performance_monitor

        monitor = get_performance_monitor()

        # Add baseline data (simulate normal operation)
        for i in range(20):
            monitor.record_gauge('test_metric', 50.0 + (i % 5))  # Normal variation

        # Add anomalous data
        monitor.record_gauge('test_metric', 200.0)  # Clear anomaly

        # Check for anomalies (this would normally run in background)
        monitor._check_for_anomalies()

        # Should have detected anomaly and created alert
        self.assertGreater(len(monitor.active_alerts), 0)

    def test_trend_analysis(self):
        """Test trend analysis functionality"""
        from core.performance_dashboard import get_performance_dashboard

        dashboard = get_performance_dashboard()

        # Generate report with trend analysis
        report = dashboard.generate_performance_report(1, include_historical=True)

        # Should include trends
        self.assertIn('trends', report)

    def test_budget_monitoring(self):
        """Test budget monitoring functionality"""
        from core.cost_analyzer import get_cost_tracker

        tracker = get_cost_tracker()

        # Set budget
        tracker.set_budget('openai', 10.0, 'monthly')

        # Add some costs
        tracker.track_api_call('openai', 'gpt-4', 1000, 500, True)

        # Check budget alerts
        alerts = tracker.check_budget_alerts()

        # Should not have alerts yet (cost is low)
        initial_alert_count = len(alerts)

        # Add more costs to trigger alert
        for _ in range(20):
            tracker.track_api_call('openai', 'gpt-4', 1000, 500, True)

        # Should now have budget alerts
        alerts = tracker.check_budget_alerts()
        self.assertGreater(len(alerts), initial_alert_count)

    def test_provider_cost_comparison(self):
        """Test provider cost comparison functionality"""
        from core.cost_analyzer import get_cost_tracker

        tracker = get_cost_tracker()

        # Add costs for different providers
        tracker.track_api_call('openai', 'gpt-4', 1000, 500, True)
        tracker.track_api_call('gemini', 'gemini-pro', 800, 200, True)

        # Compare providers
        comparison = tracker.compare_providers([
            ('openai', 'gpt-4'),
            ('gemini', 'gemini-pro')
        ])

        # Should have cost comparison data
        self.assertIn('openai/gpt-4', comparison)
        self.assertIn('gemini/gemini-pro', comparison)

    def test_resource_utilization_alerts(self):
        """Test resource utilization alerts"""
        from core.resource_monitor import get_resource_monitor

        monitor = get_resource_monitor()

        # Set custom threshold
        monitor.set_threshold('memory_percent', 10.0)  # Very low threshold

        # Get snapshot (should trigger alert due to low threshold)
        snapshot = monitor.get_current_snapshot()

        # Check if alerts were generated
        alerts = monitor._check_resource_alerts(monitor.get_resource_summary(3600))

        # Should have memory alert due to low threshold
        memory_alerts = [a for a in alerts if 'memory' in a.lower()]
        self.assertGreater(len(memory_alerts), 0)

    def test_performance_score_calculation(self):
        """Test performance score calculation"""
        from core.performance_dashboard import get_performance_dashboard

        dashboard = get_performance_dashboard()

        # Get system overview (includes performance score)
        overview = dashboard.get_system_overview()

        # Performance score should be a number between 0-100
        score = overview.get('performance_score', 0)
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

    def test_cost_forecasting(self):
        """Test cost forecasting functionality"""
        from core.cost_analyzer import get_cost_tracker

        tracker = get_cost_tracker()

        # Add historical cost data
        for i in range(10):
            tracker.track_api_call('openai', 'gpt-4', 1000, 500, True)
            time.sleep(0.01)  # Small delay to create time series

        # Generate forecast
        forecast = tracker.get_cost_forecast('openai', 30)

        # Verify forecast structure
        self.assertIn('forecasted_cost', forecast)
        self.assertIn('confidence', forecast)
        self.assertIn('daily_average', forecast)

    def test_integration_with_translation_system(self):
        """Test integration with the main translation system"""
        from core.translation import translate_batch
        from core.performance_monitor import get_performance_monitor

        # Mock translation function
        def mock_translate_fn(batch):
            time.sleep(0.1)  # Simulate API call
            return [(i, f"Translated {text}") for i, text in batch]

        # Mock batch data
        batch = [(0, "Test subtitle 1"), (1, "Test subtitle 2")]

        # Record initial metrics
        monitor = get_performance_monitor()
        initial_calls = monitor.counters.get('api_calls', 0)

        # Execute translation batch
        result = translate_batch(batch, 'es', 'test-model', 'test-key', mock_translate_fn, batch_id='test_batch')

        # Verify metrics were recorded
        self.assertGreater(monitor.counters.get('api_calls', 0), initial_calls)
        self.assertIn('test_batch', [stat.get('batch_id') for stat in monitor.batch_stats])

        # Verify result
        self.assertEqual(len(result), 2)
        self.assertTrue(all('Translated' in translation for _, translation in result))


class TestPerformanceMonitoringIntegration(unittest.TestCase):
    """Integration tests for the complete monitoring system"""

    def test_full_system_integration(self):
        """Test the complete monitoring system working together"""
        # Import all monitoring components
        from core.performance_monitor import get_performance_monitor
        from core.resource_monitor import get_resource_monitor
        from core.cost_analyzer import get_cost_tracker
        from core.performance_dashboard import get_performance_dashboard

        # Get all monitor instances
        perf_monitor = get_performance_monitor()
        resource_monitor = get_resource_monitor()
        cost_tracker = get_cost_tracker()
        dashboard = get_performance_dashboard()

        # Simulate system activity
        perf_monitor.record_counter('test_activity', 10)
        perf_monitor.record_gauge('test_performance', 85.0)

        cost_tracker.track_api_call('openai', 'gpt-4', 1000, 500, True)
        cost_tracker.track_api_call('gemini', 'gemini-pro', 800, 300, True)

        # Get comprehensive system overview
        overview = dashboard.get_system_overview()

        # Verify all components are working
        self.assertIn('performance_score', overview)
        self.assertIn('cost_efficiency', overview)
        self.assertIn('resource_utilization', overview)
        self.assertIn('recommendations', overview)

        # Generate full report
        report = dashboard.generate_performance_report(1)

        # Verify report completeness
        self.assertIn('performance_analysis', report)
        self.assertIn('cost_analysis', report)
        self.assertIn('resource_analysis', report)
        self.assertIn('insights', report)
        self.assertIn('recommendations', report)

    def test_error_handling_and_resilience(self):
        """Test error handling and system resilience"""
        from core.performance_monitor import get_performance_monitor

        monitor = get_performance_monitor()

        # Test with invalid data
        monitor.record_gauge('test', float('inf'))  # Invalid value
        monitor.record_histogram('test', float('nan'))  # Invalid value

        # System should handle errors gracefully
        summary = monitor.get_metrics_summary()

        # Should still have valid summary
        self.assertIsInstance(summary, dict)
        self.assertIn('total_metrics_collected', summary)

    def test_memory_efficiency(self):
        """Test memory efficiency of the monitoring system"""
        from core.performance_monitor import get_performance_monitor

        monitor = get_performance_monitor()

        # Record many metrics
        for i in range(1000):
            monitor.record_counter(f'counter_{i}', 1)
            monitor.record_gauge(f'gauge_{i}', float(i))
            monitor.record_histogram('histogram', float(i))
            monitor.record_timer('timer', float(i) * 0.001)

        # Check that system maintains reasonable memory usage
        summary = monitor.get_metrics_summary()

        # Should have collected metrics but not grow indefinitely
        self.assertGreater(summary['total_metrics_collected'], 0)
        self.assertLessEqual(len(monitor.metric_history), monitor.max_history_size)


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPerformanceMonitoring)
    integration_suite = unittest.TestLoader().loadTestsFromTestCase(TestPerformanceMonitoringIntegration)

    # Combine suites
    all_tests = unittest.TestSuite([suite, integration_suite])

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(all_tests)

    # Print summary
    print(f"\nTest Results: {result.testsRun} tests run")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")