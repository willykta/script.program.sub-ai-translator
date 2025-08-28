"""
Performance Dashboard and Reporting System

This module provides comprehensive performance dashboards, reports, and analytics
for monitoring translation system effectiveness and optimization progress.

Features:
- Real-time performance dashboards
- Historical performance analysis and trends
- Performance benchmarking and comparison
- Automated report generation
- Integration with existing logging system
- Export capabilities for external analysis
- Performance insights and recommendations

Expected benefits:
- Visual performance monitoring
- Data-driven optimization decisions
- Performance trend analysis
- Automated reporting and alerting
- Historical performance comparison
"""

import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import statistics

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
    log_function = lambda msg, level=0: print(f"[DASHBOARD] {msg}")

# Import monitoring components
try:
    from .performance_monitor import get_performance_monitor
    PERFORMANCE_MONITORING_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITORING_AVAILABLE = False

try:
    from .resource_monitor import get_resource_monitor
    RESOURCE_MONITORING_AVAILABLE = True
except ImportError:
    RESOURCE_MONITORING_AVAILABLE = False

try:
    from .cost_analyzer import get_cost_tracker
    COST_TRACKING_AVAILABLE = True
except ImportError:
    COST_TRACKING_AVAILABLE = False

try:
    from .connection_pool import get_connection_pool_manager
    CONNECTION_POOL_AVAILABLE = True
except ImportError:
    CONNECTION_POOL_AVAILABLE = False

try:
    from .batch_retry_optimizer import get_batch_retry_optimizer
    BATCH_RETRY_AVAILABLE = True
except ImportError:
    BATCH_RETRY_AVAILABLE = False


class PerformanceDashboard:
    """Comprehensive performance dashboard and reporting system"""

    def __init__(self):
        self.reports_dir = os.path.join(os.path.dirname(__file__), "reports")
        os.makedirs(self.reports_dir, exist_ok=True)

        # Dashboard configuration
        self.dashboard_config = {
            'update_interval_seconds': 60,
            'retention_days': 30,
            'enable_auto_reports': True,
            'report_schedule': ['daily', 'weekly', 'monthly']
        }

    def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system performance overview"""
        overview = {
            'timestamp': time.time(),
            'system_health': 'unknown',
            'active_alerts': 0,
            'performance_score': 0.0,
            'cost_efficiency': 0.0,
            'resource_utilization': {},
            'performance_metrics': {},
            'cost_metrics': {},
            'recommendations': []
        }

        # Gather data from all monitoring systems
        if PERFORMANCE_MONITORING_AVAILABLE:
            monitor = get_performance_monitor()
            summary = monitor.get_metrics_summary(3600)  # Last hour
            overview['performance_metrics'] = summary
            overview['active_alerts'] = len(monitor.active_alerts)

            # Calculate performance score (0-100)
            overview['performance_score'] = self._calculate_performance_score(summary)

        if RESOURCE_MONITORING_AVAILABLE:
            resource_monitor = get_resource_monitor()
            resource_summary = resource_monitor.get_resource_summary(3600)
            overview['resource_utilization'] = resource_summary

        if COST_TRACKING_AVAILABLE:
            cost_tracker = get_cost_tracker()
            cost_summary = cost_tracker.get_cost_summary(24)  # Last 24 hours
            overview['cost_metrics'] = cost_summary
            overview['cost_efficiency'] = self._calculate_cost_efficiency(cost_summary)

        # Determine system health
        overview['system_health'] = self._determine_system_health(overview)

        # Generate recommendations
        overview['recommendations'] = self._generate_system_recommendations(overview)

        return overview

    def _calculate_performance_score(self, metrics_summary: Dict[str, Any]) -> float:
        """Calculate overall performance score (0-100)"""
        if not metrics_summary or metrics_summary.get('total_metrics_collected', 0) == 0:
            return 50.0  # Neutral score for no data

        score = 50.0  # Base score

        # API success rate (30% weight)
        if 'timer_stats' in metrics_summary and 'api_call_duration' in metrics_summary['timer_stats']:
            api_stats = metrics_summary['timer_stats']['api_call_duration']
            # Lower is better for duration, so invert the score
            duration_score = max(0, 100 - (api_stats.get('mean', 10) * 10))
            score += (duration_score - 50) * 0.3

        # Throughput score (20% weight)
        if 'gauge_stats' in metrics_summary and 'translation_throughput' in metrics_summary.get('gauge_stats', {}):
            throughput = metrics_summary['gauge_stats']['translation_throughput'].get('mean', 0)
            throughput_score = min(100, throughput * 20)  # 5 items/sec = 100 score
            score += (throughput_score - 50) * 0.2

        # Memory efficiency (15% weight)
        if 'gauge_stats' in metrics_summary and 'memory_usage_mb' in metrics_summary.get('gauge_stats', {}):
            memory_usage = metrics_summary['gauge_stats']['memory_usage_mb'].get('mean', 500)
            memory_score = max(0, 100 - (memory_usage / 10))  # 1000MB = 0 score
            score += (memory_score - 50) * 0.15

        # Error rate penalty (20% weight)
        if 'counters' in metrics_summary:
            api_calls = metrics_summary['counters'].get('api_calls', 0)
            api_errors = metrics_summary['counters'].get('api_errors', 0)
            if api_calls > 0:
                error_rate = api_errors / api_calls
                error_penalty = error_rate * 100  # 10% error rate = 10 point penalty
                score -= error_penalty * 0.2

        # Batch efficiency (15% weight)
        if 'histogram_stats' in metrics_summary and 'batch_size' in metrics_summary['histogram_stats']:
            batch_stats = metrics_summary['histogram_stats']['batch_size']
            avg_batch_size = batch_stats.get('mean', 10)
            batch_score = min(100, avg_batch_size * 5)  # 20 items/batch = 100 score
            score += (batch_score - 50) * 0.15

        return max(0, min(100, score))

    def _calculate_cost_efficiency(self, cost_summary: Dict[str, Any]) -> float:
        """Calculate cost efficiency score (0-100)"""
        if not cost_summary or cost_summary.get('total_cost', 0) == 0:
            return 50.0

        efficiency = 50.0  # Base efficiency

        # Cost per token (40% weight)
        cost_per_token = cost_summary.get('cost_per_token', 0.01)
        if cost_per_token < 0.005:  # Very cheap
            efficiency += 25 * 0.4
        elif cost_per_token < 0.01:  # Reasonable
            efficiency += 10 * 0.4
        elif cost_per_token > 0.02:  # Expensive
            efficiency -= 20 * 0.4

        # Cost per call (30% weight)
        avg_cost_per_call = cost_summary.get('avg_cost_per_call', 0.1)
        if avg_cost_per_call < 0.05:  # Very cheap
            efficiency += 20 * 0.3
        elif avg_cost_per_call < 0.1:  # Reasonable
            efficiency += 5 * 0.3
        elif avg_cost_per_call > 0.2:  # Expensive
            efficiency -= 15 * 0.3

        # Usage volume bonus (30% weight)
        total_tokens = cost_summary.get('total_tokens', 0)
        if total_tokens > 1000000:  # High volume
            efficiency += 15 * 0.3
        elif total_tokens > 100000:  # Medium volume
            efficiency += 5 * 0.3

        return max(0, min(100, efficiency))

    def _determine_system_health(self, overview: Dict[str, Any]) -> str:
        """Determine overall system health status"""
        alerts = overview.get('active_alerts', 0)
        performance_score = overview.get('performance_score', 50)
        cost_efficiency = overview.get('cost_efficiency', 50)

        # Critical alerts or very poor performance
        if alerts > 5 or performance_score < 20 or cost_efficiency < 20:
            return 'critical'

        # Multiple alerts or poor performance
        if alerts > 2 or performance_score < 40 or cost_efficiency < 40:
            return 'warning'

        # Some alerts or moderate performance
        if alerts > 0 or performance_score < 60 or cost_efficiency < 60:
            return 'moderate'

        # Good performance with few or no alerts
        if performance_score > 80 and cost_efficiency > 70 and alerts == 0:
            return 'excellent'

        return 'good'

    def _generate_system_recommendations(self, overview: Dict[str, Any]) -> List[str]:
        """Generate system-wide recommendations"""
        recommendations = []

        performance_score = overview.get('performance_score', 50)
        cost_efficiency = overview.get('cost_efficiency', 50)
        alerts = overview.get('active_alerts', 0)

        # Performance recommendations
        if performance_score < 40:
            recommendations.append("Performance is poor. Consider optimizing batch sizes and reducing API call frequency.")
        elif performance_score < 70:
            recommendations.append("Performance could be improved. Review resource utilization and API response times.")

        # Cost recommendations
        if cost_efficiency < 40:
            recommendations.append("High translation costs detected. Consider switching to cheaper providers or optimizing usage patterns.")
        elif cost_efficiency < 70:
            recommendations.append("Cost efficiency could be improved. Review provider selection and usage patterns.")

        # Alert recommendations
        if alerts > 5:
            recommendations.append("Multiple active alerts detected. Immediate attention required to system issues.")
        elif alerts > 0:
            recommendations.append("Active alerts present. Review and resolve performance issues.")

        # Resource recommendations
        resource_util = overview.get('resource_utilization', {})
        memory_stats = resource_util.get('memory_stats', {})
        if memory_stats.get('avg_percent', 0) > 85:
            recommendations.append("High memory usage detected. Consider memory optimization or increasing system resources.")

        return recommendations

    def generate_performance_report(self, time_window_hours: int = 24,
                                  include_historical: bool = True) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'generated_at': time.time(),
            'time_window_hours': time_window_hours,
            'system_overview': self.get_system_overview(),
            'performance_analysis': {},
            'cost_analysis': {},
            'resource_analysis': {},
            'trends': {},
            'insights': [],
            'recommendations': []
        }

        # Performance analysis
        if PERFORMANCE_MONITORING_AVAILABLE:
            monitor = get_performance_monitor()
            report['performance_analysis'] = monitor.get_metrics_summary(time_window_hours * 3600)

        # Cost analysis
        if COST_TRACKING_AVAILABLE:
            cost_tracker = get_cost_tracker()
            report['cost_analysis'] = cost_tracker.get_cost_summary(time_window_hours)

        # Resource analysis
        if RESOURCE_MONITORING_AVAILABLE:
            resource_monitor = get_resource_monitor()
            report['resource_analysis'] = resource_monitor.get_resource_summary(time_window_hours * 3600)

        # Historical trends
        if include_historical:
            report['trends'] = self._analyze_historical_trends(time_window_hours)

        # Generate insights and recommendations
        report['insights'] = self._generate_performance_insights(report)
        report['recommendations'] = self._generate_performance_recommendations(report)

        return report

    def _analyze_historical_trends(self, time_window_hours: int) -> Dict[str, Any]:
        """Analyze historical performance trends"""
        trends = {
            'performance_trend': 'stable',
            'cost_trend': 'stable',
            'resource_trend': 'stable',
            'efficiency_trend': 'stable'
        }

        # Analyze performance trends
        if PERFORMANCE_MONITORING_AVAILABLE:
            monitor = get_performance_monitor()
            recent_summary = monitor.get_metrics_summary(time_window_hours * 3600)
            older_summary = monitor.get_metrics_summary(time_window_hours * 3600 * 2)

            # Compare throughput
            recent_throughput = recent_summary.get('gauge_stats', {}).get('translation_throughput', {}).get('mean', 0)
            older_throughput = older_summary.get('gauge_stats', {}).get('translation_throughput', {}).get('mean', 0)

            if recent_throughput > older_throughput * 1.1:
                trends['performance_trend'] = 'improving'
            elif recent_throughput < older_throughput * 0.9:
                trends['performance_trend'] = 'declining'

        # Analyze cost trends
        if COST_TRACKING_AVAILABLE:
            cost_tracker = get_cost_tracker()
            recent_cost = cost_tracker.get_cost_summary(time_window_hours)
            older_cost = cost_tracker.get_cost_summary(time_window_hours * 2)

            recent_cost_per_token = recent_cost.get('cost_per_token', 0)
            older_cost_per_token = older_cost.get('cost_per_token', 0)

            if recent_cost_per_token < older_cost_per_token * 0.9:
                trends['cost_trend'] = 'improving'
            elif recent_cost_per_token > older_cost_per_token * 1.1:
                trends['cost_trend'] = 'declining'

        return trends

    def _generate_performance_insights(self, report: Dict[str, Any]) -> List[str]:
        """Generate performance insights from report data"""
        insights = []

        # Performance insights
        perf_analysis = report.get('performance_analysis', {})
        if perf_analysis.get('total_metrics_collected', 0) > 0:
            avg_throughput = perf_analysis.get('gauge_stats', {}).get('translation_throughput', {}).get('mean', 0)
            if avg_throughput > 10:
                insights.append(f"Excellent throughput: {avg_throughput:.1f} items/second")
            elif avg_throughput > 5:
                insights.append(f"Good throughput: {avg_throughput:.1f} items/second")
            elif avg_throughput < 2:
                insights.append(f"Low throughput: {avg_throughput:.1f} items/second - optimization needed")

        # Cost insights
        cost_analysis = report.get('cost_analysis', {})
        if cost_analysis.get('total_cost', 0) > 0:
            cost_per_token = cost_analysis.get('cost_per_token', 0)
            if cost_per_token < 0.005:
                insights.append(f"Very cost-effective: ${cost_per_token:.4f} per 1M tokens")
            elif cost_per_token < 0.01:
                insights.append(f"Cost-effective: ${cost_per_token:.4f} per 1M tokens")
            elif cost_per_token > 0.02:
                insights.append(f"High cost: ${cost_per_token:.4f} per 1M tokens")

        # Resource insights
        resource_analysis = report.get('resource_analysis', {})
        memory_stats = resource_analysis.get('memory_stats', {})
        if memory_stats.get('avg_percent', 0) > 90:
            insights.append("Critical memory usage - immediate optimization needed")
        elif memory_stats.get('avg_percent', 0) > 80:
            insights.append("High memory usage - consider optimization")

        return insights

    def _generate_performance_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations from report data"""
        recommendations = []

        # Performance recommendations
        perf_analysis = report.get('performance_analysis', {})
        if perf_analysis.get('total_metrics_collected', 0) > 0:
            # Check for slow API calls
            timer_stats = perf_analysis.get('timer_stats', {})
            if 'api_call_duration' in timer_stats:
                avg_duration = timer_stats['api_call_duration'].get('mean', 0)
                if avg_duration > 10:
                    recommendations.append("API calls are slow. Consider timeout optimization or provider switching.")

            # Check batch sizes
            histogram_stats = perf_analysis.get('histogram_stats', {})
            if 'batch_size' in histogram_stats:
                avg_batch_size = histogram_stats['batch_size'].get('mean', 0)
                if avg_batch_size < 5:
                    recommendations.append("Small batch sizes detected. Consider increasing batch sizes for better throughput.")

        # Cost recommendations
        cost_analysis = report.get('cost_analysis', {})
        if cost_analysis.get('total_cost', 0) > 0:
            provider_breakdown = cost_analysis.get('provider_breakdown', {})
            if len(provider_breakdown) > 1:
                sorted_providers = sorted(provider_breakdown.items(), key=lambda x: x[1])
                cheapest = sorted_providers[0]
                most_expensive = sorted_providers[-1]

                if most_expensive[1] > cheapest[1] * 1.5:
                    recommendations.append(f"Consider switching from {most_expensive[0]} to {cheapest[0]} for cost savings.")

        # Resource recommendations
        resource_analysis = report.get('resource_analysis', {})
        process_stats = resource_analysis.get('process_stats', {})
        if process_stats.get('avg_threads', 0) > 30:
            recommendations.append("High thread count detected. Consider thread pool optimization.")

        return recommendations

    def create_benchmark_report(self, baseline_period_hours: int = 168,
                              comparison_period_hours: int = 24) -> Dict[str, Any]:
        """Create performance benchmark comparison report"""
        benchmark = {
            'generated_at': time.time(),
            'baseline_period_hours': baseline_period_hours,
            'comparison_period_hours': comparison_period_hours,
            'baseline_metrics': {},
            'comparison_metrics': {},
            'performance_changes': {},
            'benchmark_score': 0.0
        }

        # Get baseline metrics (last week)
        if PERFORMANCE_MONITORING_AVAILABLE:
            monitor = get_performance_monitor()
            benchmark['baseline_metrics'] = monitor.get_metrics_summary(baseline_period_hours * 3600)
            benchmark['comparison_metrics'] = monitor.get_metrics_summary(comparison_period_hours * 3600)

        # Calculate performance changes
        benchmark['performance_changes'] = self._calculate_performance_changes(
            benchmark['baseline_metrics'], benchmark['comparison_metrics']
        )

        # Calculate benchmark score
        benchmark['benchmark_score'] = self._calculate_benchmark_score(benchmark['performance_changes'])

        return benchmark

    def _calculate_performance_changes(self, baseline: Dict[str, Any],
                                     comparison: Dict[str, Any]) -> Dict[str, float]:
        """Calculate percentage changes between baseline and comparison periods"""
        changes = {}

        # Throughput change
        baseline_throughput = baseline.get('gauge_stats', {}).get('translation_throughput', {}).get('mean', 0)
        comparison_throughput = comparison.get('gauge_stats', {}).get('translation_throughput', {}).get('mean', 0)

        if baseline_throughput > 0:
            changes['throughput_change_percent'] = ((comparison_throughput - baseline_throughput) / baseline_throughput) * 100

        # API duration change
        baseline_duration = baseline.get('timer_stats', {}).get('api_call_duration', {}).get('mean', 0)
        comparison_duration = comparison.get('timer_stats', {}).get('api_call_duration', {}).get('mean', 0)

        if baseline_duration > 0:
            changes['api_duration_change_percent'] = ((comparison_duration - baseline_duration) / baseline_duration) * 100

        # Memory usage change
        baseline_memory = baseline.get('gauge_stats', {}).get('memory_usage_mb', {}).get('mean', 0)
        comparison_memory = comparison.get('gauge_stats', {}).get('memory_usage_mb', {}).get('mean', 0)

        if baseline_memory > 0:
            changes['memory_change_percent'] = ((comparison_memory - baseline_memory) / baseline_memory) * 100

        return changes

    def _calculate_benchmark_score(self, changes: Dict[str, float]) -> float:
        """Calculate benchmark score based on performance changes"""
        score = 50.0  # Base score

        # Throughput improvement (40% weight)
        throughput_change = changes.get('throughput_change_percent', 0)
        if throughput_change > 20:
            score += 20 * 0.4
        elif throughput_change > 10:
            score += 10 * 0.4
        elif throughput_change < -10:
            score -= 15 * 0.4

        # API duration improvement (30% weight)
        duration_change = changes.get('api_duration_change_percent', 0)
        if duration_change < -20:  # Significant improvement (faster)
            score += 15 * 0.3
        elif duration_change < -10:
            score += 7.5 * 0.3
        elif duration_change > 20:  # Significant degradation (slower)
            score -= 12 * 0.3

        # Memory efficiency (30% weight)
        memory_change = changes.get('memory_change_percent', 0)
        if memory_change < -15:  # Significant memory reduction
            score += 10 * 0.3
        elif memory_change > 25:  # Significant memory increase
            score -= 10 * 0.3

        return max(0, min(100, score))

    def export_dashboard_report(self, filepath: str, format: str = "json",
                              time_window_hours: int = 24) -> bool:
        """Export comprehensive dashboard report"""
        try:
            report = self.generate_performance_report(time_window_hours)

            if format.lower() == "json":
                with open(filepath, 'w') as f:
                    json.dump(report, f, indent=2)

            elif format.lower() == "html":
                self._export_html_report(report, filepath)

            log_function(f"[DASHBOARD] Report exported to {filepath}", LOG_INFO)
            return True

        except Exception as e:
            log_function(f"[DASHBOARD] Failed to export report: {str(e)}", LOG_ERROR)
            return False

    def _export_html_report(self, report: Dict[str, Any], filepath: str):
        """Export report as HTML dashboard"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sub-AI Translator Performance Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .alert {{ background: #ffebee; border-left: 4px solid #f44336; }}
                .warning {{ background: #fff3e0; border-left: 4px solid #ff9800; }}
                .success {{ background: #e8f5e8; border-left: 4px solid #4caf50; }}
                .section {{ margin: 20px 0; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Sub-AI Translator Performance Dashboard</h1>
            <p><strong>Generated:</strong> {datetime.fromtimestamp(report['generated_at']).strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Time Window:</strong> {report['time_window_hours']} hours</p>

            <div class="section">
                <h2>System Overview</h2>
                <div class="metric">
                    <strong>System Health:</strong> {report['system_overview']['system_health'].upper()}<br>
                    <strong>Performance Score:</strong> {report['system_overview']['performance_score']:.1f}/100<br>
                    <strong>Cost Efficiency:</strong> {report['system_overview']['cost_efficiency']:.1f}/100<br>
                    <strong>Active Alerts:</strong> {report['system_overview']['active_alerts']}
                </div>
            </div>

            <div class="section">
                <h2>Performance Insights</h2>
                {"".join(f"<div class='metric'>• {insight}</div>" for insight in report['insights'])}
            </div>

            <div class="section">
                <h2>Recommendations</h2>
                {"".join(f"<div class='metric'>• {rec}</div>" for rec in report['recommendations'])}
            </div>

            <div class="section">
                <h2>Performance Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th><th>Description</th></tr>
        """

        # Add performance metrics to HTML
        perf_metrics = report.get('performance_analysis', {})
        if perf_metrics.get('total_metrics_collected', 0) > 0:
            html_content += f"""
                    <tr><td>Total API Calls</td><td>{perf_metrics.get('counters', {}).get('api_calls', 0)}</td><td>Number of API calls made</td></tr>
                    <tr><td>API Errors</td><td>{perf_metrics.get('counters', {}).get('api_errors', 0)}</td><td>Number of failed API calls</td></tr>
            """

        html_content += """
                </table>
            </div>

            <div class="section">
                <h2>Cost Analysis</h2>
                <table>
                    <tr><th>Provider</th><th>Cost</th><th>Percentage</th></tr>
        """

        # Add cost breakdown to HTML
        cost_analysis = report.get('cost_analysis', {})
        provider_breakdown = cost_analysis.get('provider_breakdown', {})
        total_cost = cost_analysis.get('total_cost', 0)

        for provider, cost in provider_breakdown.items():
            percentage = (cost / total_cost * 100) if total_cost > 0 else 0
            html_content += f"<tr><td>{provider}</td><td>${cost:.2f}</td><td>{percentage:.1f}%</td></tr>"

        html_content += """
                </table>
            </div>
        </body>
        </html>
        """

        with open(filepath, 'w') as f:
            f.write(html_content)

    def log_dashboard_summary(self):
        """Log a summary of the current dashboard status"""
        overview = self.get_system_overview()

        log_function("[DASHBOARD] === Performance Dashboard Summary ===", LOG_INFO)
        log_function(f"[DASHBOARD] System Health: {overview['system_health'].upper()}", LOG_INFO)
        log_function(f"[DASHBOARD] Performance Score: {overview['performance_score']:.1f}/100", LOG_INFO)
        log_function(f"[DASHBOARD] Cost Efficiency: {overview['cost_efficiency']:.1f}/100", LOG_INFO)
        log_function(f"[DASHBOARD] Active Alerts: {overview['active_alerts']}", LOG_INFO)

        if overview['recommendations']:
            log_function("[DASHBOARD] Key Recommendations:", LOG_INFO)
            for rec in overview['recommendations'][:3]:  # Show top 3
                log_function(f"[DASHBOARD] • {rec}", LOG_INFO)


# Global dashboard instance
performance_dashboard = PerformanceDashboard()


def get_performance_dashboard() -> PerformanceDashboard:
    """Get the global performance dashboard instance"""
    return performance_dashboard


def generate_system_report(time_window_hours: int = 24) -> Dict[str, Any]:
    """Generate a system performance report"""
    return performance_dashboard.generate_performance_report(time_window_hours)


def log_system_status():
    """Log current system status"""
    performance_dashboard.log_dashboard_summary()


# Convenience functions for quick access
def get_system_health() -> str:
    """Get current system health status"""
    overview = performance_dashboard.get_system_overview()
    return overview.get('system_health', 'unknown')


def get_performance_score() -> float:
    """Get current performance score"""
    overview = performance_dashboard.get_system_overview()
    return overview.get('performance_score', 0.0)


def get_active_alerts_count() -> int:
    """Get number of active alerts"""
    overview = performance_dashboard.get_system_overview()
    return overview.get('active_alerts', 0)