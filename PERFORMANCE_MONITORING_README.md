# Comprehensive Performance Monitoring and Metrics System

This document provides complete documentation for the Sub-AI Translator's comprehensive performance monitoring and metrics system.

## Overview

The performance monitoring system provides real-time visibility into translation system effectiveness and optimization progress. It tracks metrics across all components including API calls, batch processing, resource utilization, costs, and connection pooling.

## Key Features

- **Real-time Performance Monitoring**: Continuous tracking of system performance metrics
- **Cost Analysis and Optimization**: Detailed cost tracking with optimization recommendations
- **Resource Utilization Monitoring**: Memory, CPU, disk, and network usage tracking
- **Intelligent Alerting**: Configurable alerts for performance issues and anomalies
- **Historical Analysis**: Trend analysis and performance benchmarking
- **Export Capabilities**: Data export for external analysis tools
- **Automated Recommendations**: AI-powered optimization suggestions

## Architecture

The monitoring system consists of several integrated components:

### 1. Performance Monitor (`core/performance_monitor.py`)

Central metrics collection and aggregation system.

**Key Classes:**

- `PerformanceMetricsCollector`: Main metrics collection engine
- `MetricData`: Container for metric data with metadata
- `PerformanceAlert`: Alert management system

**Features:**

- Counter, gauge, histogram, and timer metrics
- Configurable alert rules with cooldown periods
- Statistical anomaly detection (3-sigma rule)
- Automatic data cleanup and persistence

### 2. Resource Monitor (`core/resource_monitor.py`)

Comprehensive system resource utilization monitoring.

**Key Classes:**

- `ResourceMonitor`: Main resource monitoring engine
- `ResourceSnapshot`: Point-in-time resource usage snapshot

**Monitored Resources:**

- Memory usage (RAM, virtual memory, swap)
- CPU utilization and load averages
- Disk I/O statistics
- Network I/O statistics
- Process-specific metrics (threads, connections, memory)

### 3. Cost Analyzer (`core/cost_analyzer.py`)

Cost tracking and optimization analysis.

**Key Classes:**

- `CostTracker`: Cost tracking and analysis engine
- `ProviderCostProfile`: Cost profiles for different API providers

**Features:**

- Real-time cost calculation for all providers
- Budget monitoring and alerts
- Cost forecasting and optimization recommendations
- Provider cost comparison

### 4. Performance Dashboard (`core/performance_dashboard.py`)

Comprehensive reporting and dashboard system.

**Key Classes:**

- `PerformanceDashboard`: Main dashboard and reporting engine

**Features:**

- System health assessment
- Performance scoring (0-100 scale)
- Automated insights and recommendations
- Benchmark reporting and comparison
- Multiple export formats (JSON, HTML, CSV)

## Quick Start

### Basic Usage

```python
from core.performance_monitor import record_api_call, record_batch_metrics
from core.resource_monitor import get_resource_monitor
from core.cost_analyzer import get_cost_tracker
from core.performance_dashboard import get_performance_dashboard

# Record API call metrics
record_api_call('openai', 1.5, True, 1000)  # provider, duration, success, tokens

# Record batch processing metrics
record_batch_metrics(10, 5.0, True, 10, 'openai')  # batch_size, duration, success, items, provider

# Get resource usage
resource_monitor = get_resource_monitor()
current_usage = resource_monitor.get_current_snapshot()

# Track costs
cost_tracker = get_cost_tracker()
cost_tracker.track_api_call('openai', 'gpt-4', 1000, 500, True)

# Get system overview
dashboard = get_performance_dashboard()
overview = dashboard.get_system_overview()
print(f"System Health: {overview['system_health']}")
print(f"Performance Score: {overview['performance_score']:.1f}/100")
```

### Integration with Translation System

The monitoring system is automatically integrated with the translation system. No additional code is required for basic monitoring.

```python
from core.translation import translate_subtitles

# Translation with automatic monitoring
result = translate_subtitles(
    path="subtitles.srt",
    api_key="your-api-key",
    lang="es",
    model="gpt-4",
    call_fn=your_call_function
)
# Performance metrics are automatically recorded
```

## Configuration

### Alert Rules

Configure custom alert rules for proactive monitoring:

```python
from core.performance_monitor import get_performance_monitor, AlertCondition, AlertSeverity

monitor = get_performance_monitor()

# Add custom alert rule
monitor.add_alert_rule(
    'api_call_duration',                    # Metric name
    AlertCondition.ABOVE_THRESHOLD,         # Condition type
    30.0,                                  # Threshold value
    AlertSeverity.WARNING,                  # Alert severity
    "Slow API response: {value:.2f}s (threshold: {threshold}s)",  # Message template
    cooldown_seconds=300                    # Cooldown period
)
```

### Budget Monitoring

Set cost budgets for different providers:

```python
from core.cost_analyzer import get_cost_tracker

tracker = get_cost_tracker()

# Set monthly budget for OpenAI
tracker.set_budget('openai', 50.0, 'monthly')

# Set daily budget for Gemini
tracker.set_budget('gemini', 10.0, 'daily')
```

### Resource Thresholds

Configure resource usage thresholds:

```python
from core.resource_monitor import get_resource_monitor

monitor = get_resource_monitor()

# Set custom memory threshold (alert if > 90%)
monitor.set_threshold('memory_percent', 90.0)

# Set custom CPU threshold (alert if > 95%)
monitor.set_threshold('cpu_percent', 95.0)
```

## API Reference

### Performance Monitor

#### `record_counter(name, value=1, tags=None, description="")`

Record a monotonically increasing counter metric.

#### `record_gauge(name, value, tags=None, description="")`

Record a gauge metric that can increase or decrease.

#### `record_histogram(name, value, tags=None, description="")`

Record a histogram value for distribution analysis.

#### `record_timer(name, duration, tags=None, description="")`

Record a timing/duration metric.

#### `add_alert_rule(metric_name, condition, threshold, severity, message_template, cooldown_seconds=300)`

Add a custom alert rule.

#### `get_metrics_summary(time_window_seconds=3600)`

Get comprehensive metrics summary for the specified time window.

### Resource Monitor

#### `get_current_snapshot()`

Get current resource usage snapshot.

#### `get_resource_summary(time_window_seconds=3600)`

Get resource usage summary with statistics and trends.

#### `set_threshold(resource, value)`

Set alert threshold for a resource type.

### Cost Analyzer

#### `track_api_call(provider, model, input_tokens, output_tokens, success=True)`

Track an API call with cost calculation.

#### `get_cost_summary(time_window_hours=24)`

Get cost summary for the specified time window.

#### `set_budget(provider, amount, period='monthly')`

Set budget limit for a provider.

#### `get_optimization_recommendations()`

Get cost optimization recommendations.

### Performance Dashboard

#### `get_system_overview()`

Get comprehensive system performance overview.

#### `generate_performance_report(time_window_hours=24, include_historical=True)`

Generate detailed performance report.

#### `create_benchmark_report(baseline_period_hours=168, comparison_period_hours=24)`

Create performance benchmark comparison report.

#### `export_dashboard_report(filepath, format='json')`

Export dashboard report in specified format.

## Metrics Reference

### Performance Metrics

| Metric Name                  | Type      | Description                 |
| ---------------------------- | --------- | --------------------------- |
| `api_calls`                  | Counter   | Total number of API calls   |
| `api_errors`                 | Counter   | Number of failed API calls  |
| `api_call_duration`          | Timer     | API call response times     |
| `batch_size`                 | Histogram | Distribution of batch sizes |
| `batch_processing_time`      | Timer     | Batch processing durations  |
| `translation_throughput`     | Gauge     | Items processed per second  |
| `memory_usage_mb`            | Gauge     | Current memory usage        |
| `cpu_percent`                | Gauge     | Current CPU utilization     |
| `connection_pool_reuse_rate` | Gauge     | Connection pool efficiency  |

### Cost Metrics

| Metric Name               | Type    | Description             |
| ------------------------- | ------- | ----------------------- |
| `total_cost_usd`          | Counter | Total accumulated costs |
| `tokens_used`             | Counter | Total tokens consumed   |
| `cost_per_token`          | Gauge   | Average cost per token  |
| `estimated_cost_per_hour` | Gauge   | Projected hourly cost   |

### Resource Metrics

| Metric Name             | Type  | Description               |
| ----------------------- | ----- | ------------------------- |
| `system_memory_percent` | Gauge | System memory utilization |
| `system_cpu_percent`    | Gauge | System CPU utilization    |
| `process_memory_mb`     | Gauge | Process memory usage      |
| `process_threads`       | Gauge | Number of process threads |
| `disk_usage_percent`    | Gauge | Disk space utilization    |

## Alert Types

### Severity Levels

- **INFO**: Informational alerts
- **WARNING**: Warning alerts requiring attention
- **ERROR**: Error alerts requiring immediate action
- **CRITICAL**: Critical alerts requiring immediate intervention

### Condition Types

- **ABOVE_THRESHOLD**: Alert when metric exceeds threshold
- **BELOW_THRESHOLD**: Alert when metric falls below threshold
- **OUTSIDE_RANGE**: Alert when metric is outside acceptable range
- **ANOMALY_DETECTED**: Alert when statistical anomaly is detected
- **TREND_CHANGE**: Alert when performance trend changes significantly

## Export Formats

### JSON Export

Complete structured data export for programmatic analysis:

```json
{
  "export_timestamp": 1640995200.0,
  "metrics_summary": {...},
  "full_history": [...],
  "alert_history": [...],
  "performance_report": {...}
}
```

### HTML Export

Interactive dashboard for human consumption with charts and tables.

### CSV Export

Spreadsheet-compatible format for Excel/LibreOffice analysis.

## Best Practices

### 1. Monitoring Strategy

- Set up alerts for critical metrics (API errors, high costs, resource exhaustion)
- Regularly review performance reports for optimization opportunities
- Use benchmark reports to track performance improvements over time

### 2. Cost Management

- Set realistic budgets based on usage patterns
- Monitor cost per token trends for optimization opportunities
- Compare provider costs regularly for cost savings

### 3. Resource Optimization

- Monitor memory usage to prevent memory leaks
- Track CPU utilization for performance bottlenecks
- Set appropriate resource thresholds based on system capacity

### 4. Alert Management

- Configure appropriate cooldown periods to prevent alert spam
- Set up escalation procedures for critical alerts
- Regularly review and tune alert thresholds

## Troubleshooting

### Common Issues

#### High Memory Usage

```python
# Check current memory usage
from core.resource_monitor import get_resource_monitor
monitor = get_resource_monitor()
snapshot = monitor.get_current_snapshot()
print(f"Memory usage: {snapshot.memory_info.get('percent', 0):.1f}%")

# Get recommendations
from core.performance_dashboard import get_performance_dashboard
dashboard = get_performance_dashboard()
recommendations = dashboard.get_system_overview()['recommendations']
```

#### Slow API Responses

```python
# Check API performance
from core.performance_monitor import get_performance_monitor
monitor = get_performance_monitor()
timers = monitor.timers.get('api_call_duration', [])
if timers:
    avg_duration = sum(timers) / len(timers)
    print(f"Average API duration: {avg_duration:.2f}s")
```

#### High Costs

```python
# Analyze cost patterns
from core.cost_analyzer import get_cost_tracker
tracker = get_cost_tracker()
summary = tracker.get_cost_summary(24)
recommendations = tracker.get_optimization_recommendations()
```

### Performance Tuning

#### Batch Size Optimization

```python
# Analyze batch performance
from core.performance_monitor import get_performance_monitor
monitor = get_performance_monitor()

batch_sizes = monitor.histograms.get('batch_size', [])
processing_times = monitor.timers.get('batch_processing_time', [])

if batch_sizes and processing_times:
    # Calculate correlation between batch size and processing time
    import statistics
    correlation = statistics.correlation(batch_sizes, processing_times)
    print(f"Batch size/time correlation: {correlation:.3f}")
```

#### Connection Pool Tuning

```python
# Check connection pool efficiency
from core.connection_pool import get_connection_pool_manager
pool_manager = get_connection_pool_manager()
metrics = pool_manager.get_all_metrics()

for provider, provider_metrics in metrics.get('providers', {}).items():
    reuse_rate = provider_metrics.get('reuse_rate_percent', 0)
    print(f"{provider} connection reuse rate: {reuse_rate:.1f}%")
```

## Examples

### Complete Monitoring Setup

```python
#!/usr/bin/env python3
"""
Complete performance monitoring setup example
"""

from core.performance_monitor import get_performance_monitor, AlertCondition, AlertSeverity
from core.resource_monitor import get_resource_monitor
from core.cost_analyzer import get_cost_tracker
from core.performance_dashboard import get_performance_dashboard

def setup_monitoring():
    """Set up comprehensive performance monitoring"""

    # Get monitor instances
    perf_monitor = get_performance_monitor()
    resource_monitor = get_resource_monitor()
    cost_tracker = get_cost_tracker()
    dashboard = get_performance_dashboard()

    # Configure alert rules
    perf_monitor.add_alert_rule(
        'api_errors',
        AlertCondition.ABOVE_THRESHOLD,
        10,
        AlertSeverity.WARNING,
        "High API error rate: {value} errors detected"
    )

    perf_monitor.add_alert_rule(
        'api_call_duration',
        AlertCondition.ABOVE_THRESHOLD,
        30.0,
        AlertSeverity.ERROR,
        "Slow API responses: {value:.2f}s average"
    )

    # Set resource thresholds
    resource_monitor.set_threshold('memory_percent', 85.0)
    resource_monitor.set_threshold('cpu_percent', 90.0)

    # Set cost budgets
    cost_tracker.set_budget('openai', 100.0, 'monthly')
    cost_tracker.set_budget('gemini', 50.0, 'monthly')

    print("Performance monitoring setup complete!")

def generate_daily_report():
    """Generate daily performance report"""

    dashboard = get_performance_dashboard()

    # Generate comprehensive report
    report = dashboard.generate_performance_report(time_window_hours=24)

    # Export to HTML
    dashboard.export_dashboard_report('daily_report.html', 'html')

    # Export to JSON for analysis
    dashboard.export_dashboard_report('daily_report.json', 'json')

    # Log key metrics
    overview = report['system_overview']
    print("=== Daily Performance Report ===")
    print(f"System Health: {overview['system_health']}")
    print(f"Performance Score: {overview['performance_score']:.1f}/100")
    print(f"Cost Efficiency: {overview['cost_efficiency']:.1f}/100")
    print(f"Active Alerts: {overview['active_alerts']}")

    if overview['recommendations']:
        print("\nRecommendations:")
        for rec in overview['recommendations']:
            print(f"• {rec}")

if __name__ == '__main__':
    setup_monitoring()
    generate_daily_report()
```

### Custom Metrics and Alerts

```python
#!/usr/bin/env python3
"""
Custom metrics and alerting example
"""

from core.performance_monitor import get_performance_monitor, AlertCondition, AlertSeverity
import time

def custom_monitoring_example():
    """Example of custom metrics and alerting"""

    monitor = get_performance_monitor()

    # Define custom metrics
    monitor.record_counter('custom_translations_started', 0, description="Custom translation jobs started")
    monitor.record_gauge('custom_queue_length', 0, description="Length of custom processing queue")

    # Add custom alert rules
    monitor.add_alert_rule(
        'custom_queue_length',
        AlertCondition.ABOVE_THRESHOLD,
        100,
        AlertSeverity.WARNING,
        "Processing queue too long: {value} items (threshold: {threshold})"
    )

    # Simulate monitoring
    for i in range(10):
        # Record custom metrics
        monitor.record_counter('custom_translations_started')
        monitor.record_gauge('custom_queue_length', i * 10)

        # Simulate some processing
        time.sleep(0.1)

        # Check for alerts
        active_alerts = list(monitor.active_alerts.values())
        if active_alerts:
            print("Active alerts:")
            for alert in active_alerts:
                print(f"  {alert.severity.value.upper()}: {alert.message}")

    # Generate summary
    summary = monitor.get_metrics_summary()
    print(f"\nCustom metrics collected: {summary['total_metrics_collected']}")
    print(f"Active alerts: {len(monitor.active_alerts)}")

if __name__ == '__main__':
    custom_monitoring_example()
```

## Contributing

When adding new metrics or monitoring features:

1. Follow the existing naming conventions
2. Add appropriate documentation
3. Include unit tests
4. Update this documentation
5. Consider backward compatibility

## License

This performance monitoring system is part of the Sub-AI Translator project.
