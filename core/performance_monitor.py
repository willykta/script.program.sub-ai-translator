"""
Comprehensive Performance Monitoring and Metrics System

This module provides a unified performance monitoring system that tracks optimization
effectiveness and provides insights for further improvements across all components
of the Sub-AI Translator.

Features:
- Unified metrics collection from all system components
- Real-time performance monitoring and alerting
- Historical analysis and trend detection
- Cost optimization insights and recommendations
- Resource utilization tracking
- Performance benchmarking and comparison
- Export capabilities for external analysis
- Integration with existing logging system

Expected benefits:
- Real-time visibility into optimization effectiveness
- Data-driven decisions for further improvements
- Cost optimization insights
- Proactive issue detection and alerting
- Performance benchmarking and comparison
"""

import time
import threading
import statistics
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum
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
    log_function = lambda msg, level=0: print(f"[PERFORMANCE] {msg}")


class MetricType(Enum):
    """Types of performance metrics"""
    COUNTER = "counter"      # Monotonically increasing value
    GAUGE = "gauge"         # Value that can go up or down
    HISTOGRAM = "histogram" # Distribution of values
    TIMER = "timer"         # Duration measurements


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertCondition(Enum):
    """Types of alert conditions"""
    ABOVE_THRESHOLD = "above_threshold"
    BELOW_THRESHOLD = "below_threshold"
    OUTSIDE_RANGE = "outside_range"
    ANOMALY_DETECTED = "anomaly_detected"
    TREND_CHANGE = "trend_change"


class PerformanceAlert:
    """Represents a performance alert"""

    def __init__(self, alert_id: str, severity: AlertSeverity, condition: AlertCondition,
                 metric_name: str, current_value: float, threshold: float,
                 message: str, timestamp: float = None):
        self.alert_id = alert_id
        self.severity = severity
        self.condition = condition
        self.metric_name = metric_name
        self.current_value = current_value
        self.threshold = threshold
        self.message = message
        self.timestamp = timestamp or time.time()
        self.acknowledged = False
        self.resolved = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization"""
        return {
            'alert_id': self.alert_id,
            'severity': self.severity.value,
            'condition': self.condition.value,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'threshold': self.threshold,
            'message': self.message,
            'timestamp': self.timestamp,
            'acknowledged': self.acknowledged,
            'resolved': self.resolved
        }


class MetricData:
    """Container for metric data with metadata"""

    def __init__(self, name: str, metric_type: MetricType, value: float,
                 timestamp: float = None, tags: Dict[str, str] = None,
                 description: str = ""):
        self.name = name
        self.metric_type = metric_type
        self.value = value
        self.timestamp = timestamp or time.time()
        self.tags = tags or {}
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary for serialization"""
        return {
            'name': self.name,
            'type': self.metric_type.value,
            'value': self.value,
            'timestamp': self.timestamp,
            'tags': self.tags,
            'description': self.description
        }


class PerformanceMetricsCollector:
    """Unified metrics collector for all system components"""

    def __init__(self, max_history_size: int = 10000, enable_persistence: bool = True):
        self.max_history_size = max_history_size
        self.enable_persistence = enable_persistence

        # Core metrics storage
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        self.timers = defaultdict(list)

        # Historical data with timestamps
        self.metric_history = deque(maxlen=max_history_size)
        self.alert_history = deque(maxlen=1000)

        # Active alerts
        self.active_alerts = {}

        # Thread safety
        self.lock = threading.RLock()

        # Component-specific metrics collectors
        self.component_collectors = {}

        # Alert rules
        self.alert_rules = []

        # Performance baselines for anomaly detection
        self.baselines = {}

        # Start background tasks
        self._start_background_tasks()

    def _start_background_tasks(self):
        """Start background monitoring tasks"""
        def metrics_cleanup_worker():
            """Periodically clean up old metrics and check for anomalies"""
            while True:
                try:
                    time.sleep(300)  # Run every 5 minutes
                    self._cleanup_old_metrics()
                    self._check_for_anomalies()
                    self._update_baselines()
                except Exception as e:
                    log_function(f"[PERFORMANCE] Background task error: {str(e)}", LOG_WARNING)

        def persistence_worker():
            """Periodically persist metrics to disk"""
            if not self.enable_persistence:
                return

            while True:
                try:
                    time.sleep(600)  # Run every 10 minutes
                    self._persist_metrics()
                except Exception as e:
                    log_function(f"[PERFORMANCE] Persistence error: {str(e)}", LOG_WARNING)

        # Start threads
        cleanup_thread = threading.Thread(target=metrics_cleanup_worker, daemon=True)
        cleanup_thread.start()

        if self.enable_persistence:
            persistence_thread = threading.Thread(target=persistence_worker, daemon=True)
            persistence_thread.start()

    def record_counter(self, name: str, value: float = 1, tags: Dict[str, str] = None,
                      description: str = ""):
        """Record a counter metric"""
        with self.lock:
            self.counters[name] += value
            metric = MetricData(name, MetricType.COUNTER, self.counters[name],
                              tags=tags, description=description)
            self.metric_history.append(metric)
            self._check_alerts(metric)

    def record_gauge(self, name: str, value: float, tags: Dict[str, str] = None,
                    description: str = ""):
        """Record a gauge metric"""
        with self.lock:
            self.gauges[name] = value
            metric = MetricData(name, MetricType.GAUGE, value,
                              tags=tags, description=description)
            self.metric_history.append(metric)
            self._check_alerts(metric)

    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None,
                        description: str = ""):
        """Record a histogram value"""
        with self.lock:
            self.histograms[name].append(value)
            # Keep only recent values for memory efficiency
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-500:]

            metric = MetricData(name, MetricType.HISTOGRAM, value,
                              tags=tags, description=description)
            self.metric_history.append(metric)
            self._check_alerts(metric)

    def record_timer(self, name: str, duration: float, tags: Dict[str, str] = None,
                    description: str = ""):
        """Record a timer/duration metric"""
        with self.lock:
            self.timers[name].append(duration)
            # Keep only recent values
            if len(self.timers[name]) > 1000:
                self.timers[name] = self.timers[name][-500:]

            metric = MetricData(name, MetricType.TIMER, duration,
                              tags=tags, description=description)
            self.metric_history.append(metric)
            self._check_alerts(metric)

    def add_alert_rule(self, metric_name: str, condition: AlertCondition,
                      threshold: float, severity: AlertSeverity,
                      message_template: str, cooldown_seconds: int = 300):
        """Add an alert rule"""
        rule = {
            'metric_name': metric_name,
            'condition': condition,
            'threshold': threshold,
            'severity': severity,
            'message_template': message_template,
            'cooldown_seconds': cooldown_seconds,
            'last_triggered': 0
        }
        self.alert_rules.append(rule)

    def _check_alerts(self, metric: MetricData):
        """Check if any alert rules are triggered by this metric"""
        current_time = time.time()

        for rule in self.alert_rules:
            if metric.name != rule['metric_name']:
                continue

            # Check cooldown
            if current_time - rule['last_triggered'] < rule['cooldown_seconds']:
                continue

            triggered = False
            if rule['condition'] == AlertCondition.ABOVE_THRESHOLD:
                triggered = metric.value > rule['threshold']
            elif rule['condition'] == AlertCondition.BELOW_THRESHOLD:
                triggered = metric.value < rule['threshold']

            if triggered:
                rule['last_triggered'] = current_time
                alert_id = f"{metric.name}_{rule['condition'].value}_{int(current_time)}"

                message = rule['message_template'].format(
                    metric_name=metric.name,
                    value=metric.value,
                    threshold=rule['threshold']
                )

                alert = PerformanceAlert(
                    alert_id=alert_id,
                    severity=rule['severity'],
                    condition=rule['condition'],
                    metric_name=metric.name,
                    current_value=metric.value,
                    threshold=rule['threshold'],
                    message=message
                )

                self.active_alerts[alert_id] = alert
                self.alert_history.append(alert)

                log_function(f"[ALERT] {alert.message}", LOG_WARNING)

    def _check_for_anomalies(self):
        """Check for performance anomalies using statistical analysis"""
        with self.lock:
            # Analyze recent metrics for anomalies
            recent_metrics = list(self.metric_history)[-100:]  # Last 100 metrics

            if len(recent_metrics) < 20:
                return

            # Group by metric name
            metric_groups = defaultdict(list)
            for metric in recent_metrics:
                metric_groups[metric.name].append(metric.value)

            # Check for anomalies in each metric group
            for name, values in metric_groups.items():
                if len(values) < 10:
                    continue

                try:
                    mean = statistics.mean(values)
                    stdev = statistics.stdev(values) if len(values) > 1 else 0

                    if stdev > 0:
                        # Check last few values for anomalies (3-sigma rule)
                        recent_values = values[-5:]
                        for i, value in enumerate(recent_values):
                            z_score = abs(value - mean) / stdev
                            if z_score > 3.0:  # 3-sigma anomaly
                                alert_id = f"anomaly_{name}_{int(time.time())}_{i}"
                                message = f"Anomaly detected in {name}: value {value:.2f} is {z_score:.1f} standard deviations from mean {mean:.2f}"

                                alert = PerformanceAlert(
                                    alert_id=alert_id,
                                    severity=AlertSeverity.WARNING,
                                    condition=AlertCondition.ANOMALY_DETECTED,
                                    metric_name=name,
                                    current_value=value,
                                    threshold=mean + 3 * stdev,
                                    message=message
                                )

                                self.active_alerts[alert_id] = alert
                                self.alert_history.append(alert)

                                log_function(f"[ANOMALY] {message}", LOG_WARNING)
                                break  # Only alert once per metric per check

                except Exception as e:
                    log_function(f"[PERFORMANCE] Anomaly detection error for {name}: {str(e)}", LOG_DEBUG)

    def _update_baselines(self):
        """Update performance baselines for trend analysis"""
        with self.lock:
            # Calculate baselines from recent history
            recent_window = timedelta(hours=1)
            cutoff_time = time.time() - recent_window.total_seconds()

            recent_metrics = [m for m in self.metric_history if m.timestamp > cutoff_time]

            if len(recent_metrics) < 10:
                return

            # Group by metric name and calculate baselines
            metric_groups = defaultdict(list)
            for metric in recent_metrics:
                metric_groups[metric.name].append(metric.value)

            for name, values in metric_groups.items():
                if len(values) >= 5:
                    try:
                        baseline = {
                            'mean': statistics.mean(values),
                            'median': statistics.median(values),
                            'stdev': statistics.stdev(values) if len(values) > 1 else 0,
                            'min': min(values),
                            'max': max(values),
                            'count': len(values),
                            'last_updated': time.time()
                        }
                        self.baselines[name] = baseline
                    except Exception as e:
                        log_function(f"[PERFORMANCE] Baseline calculation error for {name}: {str(e)}", LOG_DEBUG)

    def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory bloat"""
        with self.lock:
            # Keep only recent metrics (last 24 hours)
            cutoff_time = time.time() - (24 * 60 * 60)
            original_count = len(self.metric_history)

            # Filter metric history
            self.metric_history = deque(
                [m for m in self.metric_history if m.timestamp > cutoff_time],
                maxlen=self.max_history_size
            )

            # Clean up old histogram and timer data
            for name in list(self.histograms.keys()):
                self.histograms[name] = [v for v in self.histograms[name][-1000:] if v > cutoff_time - 3600]

            for name in list(self.timers.keys()):
                self.timers[name] = [v for v in self.timers[name][-1000:] if v > cutoff_time - 3600]

            cleaned_count = original_count - len(self.metric_history)
            if cleaned_count > 0:
                log_function(f"[PERFORMANCE] Cleaned up {cleaned_count} old metrics", LOG_DEBUG)

    def _persist_metrics(self):
        """Persist metrics to disk for historical analysis"""
        try:
            # Create metrics directory if it doesn't exist
            metrics_dir = os.path.join(os.path.dirname(__file__), "metrics")
            os.makedirs(metrics_dir, exist_ok=True)

            # Save current metrics snapshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_snapshot_{timestamp}.json"
            filepath = os.path.join(metrics_dir, filename)

            snapshot = {
                'timestamp': time.time(),
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'baselines': self.baselines,
                'active_alerts': {aid: alert.to_dict() for aid, alert in self.active_alerts.items()},
                'recent_metrics': [m.to_dict() for m in list(self.metric_history)[-100:]]
            }

            with open(filepath, 'w') as f:
                json.dump(snapshot, f, indent=2)

            # Keep only last 10 snapshots
            self._cleanup_old_snapshots(metrics_dir)

        except Exception as e:
            log_function(f"[PERFORMANCE] Failed to persist metrics: {str(e)}", LOG_WARNING)

    def _cleanup_old_snapshots(self, metrics_dir: str, max_snapshots: int = 10):
        """Clean up old metric snapshots"""
        try:
            files = [f for f in os.listdir(metrics_dir) if f.startswith("metrics_snapshot_")]
            if len(files) > max_snapshots:
                files.sort(reverse=True)  # Newest first
                files_to_delete = files[max_snapshots:]
                for filename in files_to_delete:
                    os.remove(os.path.join(metrics_dir, filename))
        except Exception as e:
            log_function(f"[PERFORMANCE] Failed to cleanup snapshots: {str(e)}", LOG_DEBUG)

    def get_metrics_summary(self, time_window_seconds: int = 3600) -> Dict[str, Any]:
        """Get a comprehensive metrics summary"""
        with self.lock:
            cutoff_time = time.time() - time_window_seconds
            recent_metrics = [m for m in self.metric_history if m.timestamp > cutoff_time]

            summary = {
                'time_window_seconds': time_window_seconds,
                'total_metrics_collected': len(self.metric_history),
                'recent_metrics_count': len(recent_metrics),
                'active_alerts_count': len(self.active_alerts),
                'total_alerts_history': len(self.alert_history),
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'baselines': self.baselines,
                'metric_types': {
                    'counters': len(self.counters),
                    'gauges': len(self.gauges),
                    'histograms': len(self.histograms),
                    'timers': len(self.timers)
                }
            }

            # Add histogram and timer statistics
            summary['histogram_stats'] = {}
            for name, values in self.histograms.items():
                if values:
                    summary['histogram_stats'][name] = {
                        'count': len(values),
                        'mean': statistics.mean(values),
                        'median': statistics.median(values),
                        'min': min(values),
                        'max': max(values)
                    }

            summary['timer_stats'] = {}
            for name, values in self.timers.items():
                if values:
                    summary['timer_stats'][name] = {
                        'count': len(values),
                        'mean': statistics.mean(values),
                        'median': statistics.median(values),
                        'p95': sorted(values)[int(len(values) * 0.95)] if len(values) > 1 else max(values),
                        'min': min(values),
                        'max': max(values)
                    }

            return summary

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report"""
        summary = self.get_metrics_summary()

        report = {
            'generated_at': time.time(),
            'summary': summary,
            'active_alerts': [alert.to_dict() for alert in self.active_alerts.values()],
            'recent_alerts': [alert.to_dict() for alert in list(self.alert_history)[-10:]],
            'performance_insights': self._generate_insights(),
            'recommendations': self._generate_recommendations()
        }

        return report

    def _generate_insights(self) -> List[str]:
        """Generate performance insights based on collected metrics"""
        insights = []

        # Analyze throughput trends
        if 'translation_throughput' in self.timers:
            throughput_values = self.timers['translation_throughput'][-20:]  # Last 20 measurements
            if len(throughput_values) >= 5:
                recent_avg = statistics.mean(throughput_values[-5:])
                older_avg = statistics.mean(throughput_values[:-5]) if len(throughput_values) > 5 else recent_avg

                if recent_avg > older_avg * 1.1:
                    insights.append(f"Translation throughput improving: {recent_avg:.1f} items/sec vs {older_avg:.1f} items/sec")
                elif recent_avg < older_avg * 0.9:
                    insights.append(f"Translation throughput declining: {recent_avg:.1f} items/sec vs {older_avg:.1f} items/sec")
        # Analyze error rates
        if 'api_errors' in self.counters and 'api_calls' in self.counters:
            error_rate = self.counters['api_errors'] / max(self.counters['api_calls'], 1)
            if error_rate > 0.1:  # More than 10% error rate
                insights.append(f"High API error rate detected: {error_rate:.1%}")
            elif error_rate < 0.01:  # Less than 1% error rate
                insights.append(f"Excellent API reliability: {error_rate:.2%} error rate")

        # Analyze memory usage trends
        if 'memory_usage_mb' in self.gauges:
            memory_usage = self.gauges['memory_usage_mb']
            if memory_usage > 500:  # High memory usage
                insights.append(f"High memory usage detected: {memory_usage:.0f}MB")
            elif memory_usage > 200:  # Moderate memory usage
                insights.append(f"Moderate memory usage: {memory_usage:.0f}MB")

        # Analyze connection pool effectiveness
        if 'connection_pool_reuse_rate' in self.gauges:
            reuse_rate = self.gauges['connection_pool_reuse_rate']
            if reuse_rate < 0.5:  # Low reuse rate
                insights.append(f"Low connection pool reuse rate: {reuse_rate:.1%} - consider increasing pool size")
            elif reuse_rate > 0.9:  # High reuse rate
                insights.append(f"Excellent connection pool reuse rate: {reuse_rate:.1%}")

        return insights

    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on metrics"""
        recommendations = []

        # Batch size optimization
        if 'batch_processing_time' in self.timers and 'batch_size' in self.histograms:
            batch_times = self.timers['batch_processing_time'][-10:]
            batch_sizes = self.histograms['batch_size'][-10:]

            if len(batch_times) >= 5 and len(batch_sizes) >= 5:
                # Calculate correlation between batch size and processing time
                try:
                    correlation = statistics.correlation(batch_sizes, batch_times)
                    if correlation > 0.7:  # Strong positive correlation
                        recommendations.append("Consider reducing batch sizes - processing time increases significantly with larger batches")
                    elif correlation < -0.3:  # Negative correlation (unusual)
                        recommendations.append("Batch size optimization opportunity detected - larger batches may be more efficient")
                except:
                    pass

        # API call optimization
        if 'api_call_duration' in self.timers:
            durations = self.timers['api_call_duration'][-20:]
            if len(durations) >= 10:
                p95_duration = sorted(durations)[int(len(durations) * 0.95)]
                if p95_duration > 30:  # P95 over 30 seconds
                    recommendations.append("API calls are slow (P95 > 30s) - consider timeout optimization or provider switching")
                elif p95_duration > 10:  # P95 over 10 seconds
                    recommendations.append("API calls could be faster (P95 > 10s) - consider connection pooling or parallel processing")

        # Cost optimization
        if 'estimated_cost_per_hour' in self.gauges:
            cost_per_hour = self.gauges['estimated_cost_per_hour']
            if cost_per_hour > 10:  # High cost
                recommendations.append(f"High translation cost detected: ${cost_per_hour:.2f}/hour - consider batch optimization or cheaper provider")
            elif cost_per_hour > 5:  # Moderate cost
                recommendations.append(f"Moderate translation cost: ${cost_per_hour:.2f}/hour - monitor for optimization opportunities")

        # Resource optimization
        if 'thread_count' in self.gauges:
            thread_count = self.gauges['thread_count']
            if thread_count > 20:  # High thread count
                recommendations.append(f"High thread count detected: {thread_count} - consider thread pool optimization")
            elif thread_count < 2:  # Low thread utilization
                recommendations.append(f"Low thread utilization: {thread_count} - consider increasing parallel processing")

        return recommendations

    def export_metrics(self, filepath: str, format: str = "json") -> bool:
        """Export metrics data for external analysis"""
        try:
            if format.lower() == "json":
                data = {
                    'export_timestamp': time.time(),
                    'metrics_summary': self.get_metrics_summary(),
                    'full_history': [m.to_dict() for m in self.metric_history],
                    'alert_history': [a.to_dict() for a in self.alert_history],
                    'performance_report': self.get_performance_report()
                }

                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)

            elif format.lower() == "csv":
                # Export metrics as CSV for spreadsheet analysis
                import csv

                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'metric_name', 'type', 'value', 'tags'])

                    for metric in self.metric_history:
                        tags_str = json.dumps(metric.tags) if metric.tags else ""
                        writer.writerow([
                            metric.timestamp,
                            metric.name,
                            metric.metric_type.value,
                            metric.value,
                            tags_str
                        ])

            log_function(f"[PERFORMANCE] Metrics exported to {filepath}", LOG_INFO)
            return True

        except Exception as e:
            log_function(f"[PERFORMANCE] Failed to export metrics: {str(e)}", LOG_ERROR)
            return False

    def reset_metrics(self):
        """Reset all metrics (useful for testing or fresh starts)"""
        with self.lock:
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
            self.timers.clear()
            self.metric_history.clear()
            self.alert_history.clear()
            self.active_alerts.clear()
            self.baselines.clear()

            log_function("[PERFORMANCE] All metrics reset", LOG_INFO)


# Global performance monitor instance
performance_monitor = PerformanceMetricsCollector()


def get_performance_monitor() -> PerformanceMetricsCollector:
    """Get the global performance monitor instance"""
    return performance_monitor


# Convenience functions for recording common metrics
def record_api_call(provider: str, duration: float, success: bool, tokens_used: int = 0):
    """Record API call metrics"""
    tags = {'provider': provider, 'success': str(success)}

    performance_monitor.record_counter('api_calls', tags=tags)
    if not success:
        performance_monitor.record_counter('api_errors', tags=tags)

    performance_monitor.record_timer('api_call_duration', duration, tags=tags)

    if tokens_used > 0:
        performance_monitor.record_counter('tokens_used', tokens_used, tags=tags)


def record_batch_metrics(batch_size: int, processing_time: float, success: bool,
                        items_processed: int, provider: str):
    """Record batch processing metrics"""
    tags = {'provider': provider, 'success': str(success)}

    performance_monitor.record_histogram('batch_size', batch_size, tags=tags)
    performance_monitor.record_timer('batch_processing_time', processing_time, tags=tags)
    performance_monitor.record_counter('batches_processed', tags=tags)

    if success:
        throughput = items_processed / processing_time if processing_time > 0 else 0
        performance_monitor.record_gauge('translation_throughput', throughput, tags=tags)


def record_resource_usage(memory_mb: float, thread_count: int, connection_count: int):
    """Record resource utilization metrics"""
    performance_monitor.record_gauge('memory_usage_mb', memory_mb)
    performance_monitor.record_gauge('thread_count', thread_count)
    performance_monitor.record_gauge('connection_count', connection_count)


def record_cost_metrics(cost_usd: float, tokens_used: int, provider: str):
    """Record cost-related metrics"""
    tags = {'provider': provider}

    performance_monitor.record_counter('total_cost_usd', cost_usd, tags=tags)
    performance_monitor.record_counter('total_tokens_used', tokens_used, tags=tags)

    # Calculate cost per token if tokens were used
    if tokens_used > 0:
        cost_per_token = cost_usd / tokens_used
        performance_monitor.record_gauge('cost_per_token', cost_per_token, tags=tags)


# Initialize default alert rules
def initialize_default_alerts():
    """Set up default performance alert rules"""
    monitor = get_performance_monitor()

    # API error rate alerts
    monitor.add_alert_rule(
        'api_errors', AlertCondition.ABOVE_THRESHOLD, 10,
        AlertSeverity.WARNING,
        "High API error rate: {value} errors detected (threshold: {threshold})"
    )

    # Slow API response alerts
    monitor.add_alert_rule(
        'api_call_duration', AlertCondition.ABOVE_THRESHOLD, 30.0,
        AlertSeverity.WARNING,
        "Slow API responses detected: {value:.2f}s average (threshold: {threshold}s)"
    )

    # High memory usage alerts
    monitor.add_alert_rule(
        'memory_usage_mb', AlertCondition.ABOVE_THRESHOLD, 1000.0,
        AlertSeverity.ERROR,
        "High memory usage: {value:.0f}MB (threshold: {threshold}MB)"
    )

    # Low throughput alerts
    monitor.add_alert_rule(
        'translation_throughput', AlertCondition.BELOW_THRESHOLD, 1.0,
        AlertSeverity.WARNING,
        "Low translation throughput: {value:.2f} items/sec (threshold: {threshold})"
    )

    log_function("[PERFORMANCE] Default alert rules initialized", LOG_INFO)


# Initialize alerts when module is imported
initialize_default_alerts()