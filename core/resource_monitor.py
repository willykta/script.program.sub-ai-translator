"""
Resource Utilization Monitoring System

This module provides comprehensive monitoring of system resources including:
- Memory usage (RAM, virtual memory)
- CPU utilization and load
- Thread and process information
- Disk I/O statistics
- Network I/O statistics
- System load averages

Features:
- Real-time resource monitoring
- Historical resource tracking
- Resource usage alerts and thresholds
- Performance correlation analysis
- Resource optimization recommendations
- Cross-platform compatibility

Expected benefits:
- Identify resource bottlenecks
- Optimize resource allocation
- Prevent resource exhaustion
- Monitor system health
- Performance troubleshooting
"""

import time
import threading
import psutil
import platform
import os
from typing import Dict, List, Any, Optional, Callable
from collections import deque
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
    log_function = lambda msg, level=0: print(f"[RESOURCE] {msg}")

# Import performance monitoring
try:
    from .performance_monitor import get_performance_monitor, record_resource_usage
    PERFORMANCE_MONITORING_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITORING_AVAILABLE = False


class ResourceSnapshot:
    """Snapshot of system resource usage at a specific point in time"""

    def __init__(self):
        self.timestamp = time.time()
        self.memory_info = self._get_memory_info()
        self.cpu_info = self._get_cpu_info()
        self.disk_info = self._get_disk_info()
        self.network_info = self._get_network_info()
        self.process_info = self._get_process_info()
        self.system_info = self._get_system_info()

    def _get_memory_info(self) -> Dict[str, Any]:
        """Get comprehensive memory information"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            return {
                'total_mb': memory.total / 1024 / 1024,
                'available_mb': memory.available / 1024 / 1024,
                'used_mb': memory.used / 1024 / 1024,
                'free_mb': memory.free / 1024 / 1024,
                'percent': memory.percent,
                'swap_total_mb': swap.total / 1024 / 1024,
                'swap_used_mb': swap.used / 1024 / 1024,
                'swap_free_mb': swap.free / 1024 / 1024,
                'swap_percent': swap.percent
            }
        except Exception as e:
            log_function(f"[RESOURCE] Failed to get memory info: {str(e)}", LOG_DEBUG)
            return {}

    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get comprehensive CPU information"""
        try:
            return {
                'percent': psutil.cpu_percent(interval=None),
                'count_logical': psutil.cpu_count(),
                'count_physical': psutil.cpu_count(logical=False),
                'load_avg': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
                'freq_current': psutil.cpu_freq().current if psutil.cpu_freq() else None,
                'freq_min': psutil.cpu_freq().min if psutil.cpu_freq() else None,
                'freq_max': psutil.cpu_freq().max if psutil.cpu_freq() else None
            }
        except Exception as e:
            log_function(f"[RESOURCE] Failed to get CPU info: {str(e)}", LOG_DEBUG)
            return {}

    def _get_disk_info(self) -> Dict[str, Any]:
        """Get disk I/O information"""
        try:
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()

            info = {
                'usage_total_gb': disk_usage.total / 1024 / 1024 / 1024,
                'usage_used_gb': disk_usage.used / 1024 / 1024 / 1024,
                'usage_free_gb': disk_usage.free / 1024 / 1024 / 1024,
                'usage_percent': disk_usage.percent
            }

            if disk_io:
                info.update({
                    'io_read_count': disk_io.read_count,
                    'io_write_count': disk_io.write_count,
                    'io_read_bytes': disk_io.read_bytes,
                    'io_write_bytes': disk_io.write_bytes,
                    'io_read_time_ms': disk_io.read_time,
                    'io_write_time_ms': disk_io.write_time
                })

            return info
        except Exception as e:
            log_function(f"[RESOURCE] Failed to get disk info: {str(e)}", LOG_DEBUG)
            return {}

    def _get_network_info(self) -> Dict[str, Any]:
        """Get network I/O information"""
        try:
            net_io = psutil.net_io_counters()

            if net_io:
                return {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv,
                    'errin': net_io.errin,
                    'errout': net_io.errout,
                    'dropin': net_io.dropin,
                    'dropout': net_io.dropout
                }
            return {}
        except Exception as e:
            log_function(f"[RESOURCE] Failed to get network info: {str(e)}", LOG_DEBUG)
            return {}

    def _get_process_info(self) -> Dict[str, Any]:
        """Get current process information"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            cpu_times = process.cpu_times()

            return {
                'pid': process.pid,
                'name': process.name(),
                'status': process.status(),
                'cpu_percent': process.cpu_percent(),
                'memory_rss_mb': memory_info.rss / 1024 / 1024,
                'memory_vms_mb': memory_info.vms / 1024 / 1024,
                'memory_percent': process.memory_percent(),
                'threads': process.num_threads(),
                'cpu_times_user': cpu_times.user,
                'cpu_times_system': cpu_times.system,
                'cpu_times_total': cpu_times.user + cpu_times.system,
                'open_files': len(process.open_files()) if hasattr(process, 'open_files') else 0,
                'connections': len(process.connections()) if hasattr(process, 'connections') else 0
            }
        except Exception as e:
            log_function(f"[RESOURCE] Failed to get process info: {str(e)}", LOG_DEBUG)
            return {}

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            return {
                'platform': platform.platform(),
                'processor': platform.processor(),
                'architecture': platform.architecture(),
                'python_version': platform.python_version(),
                'boot_time': psutil.boot_time(),
                'uptime_seconds': time.time() - psutil.boot_time()
            }
        except Exception as e:
            log_function(f"[RESOURCE] Failed to get system info: {str(e)}", LOG_DEBUG)
            return {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary"""
        return {
            'timestamp': self.timestamp,
            'memory': self.memory_info,
            'cpu': self.cpu_info,
            'disk': self.disk_info,
            'network': self.network_info,
            'process': self.process_info,
            'system': self.system_info
        }


class ResourceMonitor:
    """Comprehensive resource monitoring system"""

    def __init__(self, history_size: int = 1000, monitoring_interval: float = 5.0):
        self.history_size = history_size
        self.monitoring_interval = monitoring_interval
        self.snapshots = deque(maxlen=history_size)
        self.is_monitoring = False
        self.monitor_thread = None
        self.lock = threading.Lock()

        # Resource thresholds for alerting
        self.thresholds = {
            'memory_percent': 85.0,  # Alert if memory usage > 85%
            'cpu_percent': 90.0,     # Alert if CPU usage > 90%
            'disk_percent': 95.0,    # Alert if disk usage > 95%
        }

        # Start monitoring
        self.start_monitoring()

    def start_monitoring(self):
        """Start the resource monitoring thread"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        log_function("[RESOURCE] Resource monitoring started", LOG_INFO)

    def stop_monitoring(self):
        """Stop the resource monitoring thread"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        log_function("[RESOURCE] Resource monitoring stopped", LOG_INFO)

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                snapshot = ResourceSnapshot()

                with self.lock:
                    self.snapshots.append(snapshot)

                # Record in performance monitor if available
                if PERFORMANCE_MONITORING_AVAILABLE:
                    self._record_performance_metrics(snapshot)

                # Check thresholds and generate alerts
                self._check_thresholds(snapshot)

                time.sleep(self.monitoring_interval)

            except Exception as e:
                log_function(f"[RESOURCE] Monitoring error: {str(e)}", LOG_ERROR)
                time.sleep(self.monitoring_interval)

    def _record_performance_metrics(self, snapshot: ResourceSnapshot):
        """Record resource metrics in the performance monitor"""
        try:
            monitor = get_performance_monitor()

            # Memory metrics
            if snapshot.memory_info:
                monitor.record_gauge('system_memory_percent', snapshot.memory_info.get('percent', 0))
                monitor.record_gauge('system_memory_used_mb', snapshot.memory_info.get('used_mb', 0))
                monitor.record_gauge('process_memory_mb', snapshot.process_info.get('memory_rss_mb', 0))

            # CPU metrics
            if snapshot.cpu_info:
                monitor.record_gauge('system_cpu_percent', snapshot.cpu_info.get('percent', 0))
                monitor.record_gauge('process_cpu_percent', snapshot.process_info.get('cpu_percent', 0))

            # Thread and connection metrics
            if snapshot.process_info:
                monitor.record_gauge('process_threads', snapshot.process_info.get('threads', 0))
                monitor.record_gauge('process_connections', snapshot.process_info.get('connections', 0))

            # Disk metrics
            if snapshot.disk_info:
                monitor.record_gauge('disk_usage_percent', snapshot.disk_info.get('usage_percent', 0))

        except Exception as e:
            log_function(f"[RESOURCE] Failed to record performance metrics: {str(e)}", LOG_DEBUG)

    def _check_thresholds(self, snapshot: ResourceSnapshot):
        """Check resource usage against thresholds and generate alerts"""
        if not PERFORMANCE_MONITORING_AVAILABLE:
            return

        try:
            monitor = get_performance_monitor()

            # Memory threshold check
            memory_percent = snapshot.memory_info.get('percent', 0)
            if memory_percent > self.thresholds['memory_percent']:
                monitor.record_gauge('memory_threshold_exceeded', memory_percent)
                log_function(f"[RESOURCE] Memory usage high: {memory_percent:.1f}%", LOG_WARNING)

            # CPU threshold check
            cpu_percent = snapshot.cpu_info.get('percent', 0)
            if cpu_percent > self.thresholds['cpu_percent']:
                monitor.record_gauge('cpu_threshold_exceeded', cpu_percent)
                log_function(f"[RESOURCE] CPU usage high: {cpu_percent:.1f}%", LOG_WARNING)

            # Disk threshold check
            disk_percent = snapshot.disk_info.get('usage_percent', 0)
            if disk_percent > self.thresholds['disk_percent']:
                monitor.record_gauge('disk_threshold_exceeded', disk_percent)
                log_function(f"[RESOURCE] Disk usage high: {disk_percent:.1f}%", LOG_WARNING)

        except Exception as e:
            log_function(f"[RESOURCE] Threshold check error: {str(e)}", LOG_DEBUG)

    def get_current_snapshot(self) -> Optional[ResourceSnapshot]:
        """Get the most recent resource snapshot"""
        with self.lock:
            return self.snapshots[-1] if self.snapshots else None

    def get_resource_history(self, time_window_seconds: int = 3600) -> List[ResourceSnapshot]:
        """Get resource snapshots within the specified time window"""
        cutoff_time = time.time() - time_window_seconds

        with self.lock:
            return [s for s in self.snapshots if s.timestamp > cutoff_time]

    def get_resource_summary(self, time_window_seconds: int = 3600) -> Dict[str, Any]:
        """Get a summary of resource usage over the specified time window"""
        snapshots = self.get_resource_history(time_window_seconds)

        if not snapshots:
            return {}

        summary = {
            'time_window_seconds': time_window_seconds,
            'snapshot_count': len(snapshots),
            'memory_stats': self._calculate_memory_stats(snapshots),
            'cpu_stats': self._calculate_cpu_stats(snapshots),
            'disk_stats': self._calculate_disk_stats(snapshots),
            'network_stats': self._calculate_network_stats(snapshots),
            'process_stats': self._calculate_process_stats(snapshots)
        }

        return summary

    def _calculate_memory_stats(self, snapshots: List[ResourceSnapshot]) -> Dict[str, Any]:
        """Calculate memory usage statistics"""
        memory_percents = [s.memory_info.get('percent', 0) for s in snapshots if s.memory_info]
        memory_used_mb = [s.memory_info.get('used_mb', 0) for s in snapshots if s.memory_info]

        if not memory_percents:
            return {}

        return {
            'avg_percent': statistics.mean(memory_percents),
            'max_percent': max(memory_percents),
            'min_percent': min(memory_percents),
            'avg_used_mb': statistics.mean(memory_used_mb) if memory_used_mb else 0,
            'max_used_mb': max(memory_used_mb) if memory_used_mb else 0,
            'trend': self._calculate_trend(memory_percents)
        }

    def _calculate_cpu_stats(self, snapshots: List[ResourceSnapshot]) -> Dict[str, Any]:
        """Calculate CPU usage statistics"""
        cpu_percents = [s.cpu_info.get('percent', 0) for s in snapshots if s.cpu_info]

        if not cpu_percents:
            return {}

        return {
            'avg_percent': statistics.mean(cpu_percents),
            'max_percent': max(cpu_percents),
            'min_percent': min(cpu_percents),
            'trend': self._calculate_trend(cpu_percents)
        }

    def _calculate_disk_stats(self, snapshots: List[ResourceSnapshot]) -> Dict[str, Any]:
        """Calculate disk usage statistics"""
        disk_percents = [s.disk_info.get('usage_percent', 0) for s in snapshots if s.disk_info]

        if not disk_percents:
            return {}

        return {
            'avg_percent': statistics.mean(disk_percents),
            'max_percent': max(disk_percents),
            'min_percent': min(disk_percents),
            'trend': self._calculate_trend(disk_percents)
        }

    def _calculate_network_stats(self, snapshots: List[ResourceSnapshot]) -> Dict[str, Any]:
        """Calculate network I/O statistics"""
        if len(snapshots) < 2:
            return {}

        try:
            bytes_sent = [s.network_info.get('bytes_sent', 0) for s in snapshots if s.network_info]
            bytes_recv = [s.network_info.get('bytes_recv', 0) for s in snapshots if s.network_info]

            if len(bytes_sent) >= 2 and len(bytes_recv) >= 2:
                # Calculate rates (bytes per second)
                time_diffs = [snapshots[i+1].timestamp - snapshots[i].timestamp
                             for i in range(len(snapshots)-1)]

                sent_rates = [(bytes_sent[i+1] - bytes_sent[i]) / time_diffs[i]
                             for i in range(len(time_diffs))]
                recv_rates = [(bytes_recv[i+1] - bytes_recv[i]) / time_diffs[i]
                             for i in range(len(time_diffs))]

                return {
                    'avg_sent_rate_bps': statistics.mean(sent_rates) if sent_rates else 0,
                    'avg_recv_rate_bps': statistics.mean(recv_rates) if recv_rates else 0,
                    'max_sent_rate_bps': max(sent_rates) if sent_rates else 0,
                    'max_recv_rate_bps': max(recv_rates) if recv_rates else 0
                }
        except Exception as e:
            log_function(f"[RESOURCE] Network stats calculation error: {str(e)}", LOG_DEBUG)

        return {}

    def _calculate_process_stats(self, snapshots: List[ResourceSnapshot]) -> Dict[str, Any]:
        """Calculate process statistics"""
        thread_counts = [s.process_info.get('threads', 0) for s in snapshots if s.process_info]
        memory_mb = [s.process_info.get('memory_rss_mb', 0) for s in snapshots if s.process_info]

        if not thread_counts:
            return {}

        return {
            'avg_threads': statistics.mean(thread_counts),
            'max_threads': max(thread_counts),
            'min_threads': min(thread_counts),
            'avg_memory_mb': statistics.mean(memory_mb) if memory_mb else 0,
            'max_memory_mb': max(memory_mb) if memory_mb else 0,
            'thread_trend': self._calculate_trend(thread_counts),
            'memory_trend': self._calculate_trend(memory_mb) if memory_mb else 'stable'
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a series of values"""
        if len(values) < 3:
            return 'stable'

        try:
            # Simple linear trend calculation
            n = len(values)
            x = list(range(n))
            slope = statistics.linear_regression(x, values)[0]

            if slope > 0.1:
                return 'increasing'
            elif slope < -0.1:
                return 'decreasing'
            else:
                return 'stable'
        except:
            return 'stable'

    def get_resource_report(self) -> Dict[str, Any]:
        """Generate a comprehensive resource utilization report"""
        current = self.get_current_snapshot()
        summary = self.get_resource_summary()

        report = {
            'generated_at': time.time(),
            'current_snapshot': current.to_dict() if current else {},
            'summary': summary,
            'recommendations': self._generate_resource_recommendations(summary),
            'alerts': self._check_resource_alerts(summary)
        }

        return report

    def _generate_resource_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate resource optimization recommendations"""
        recommendations = []

        # Memory recommendations
        memory_stats = summary.get('memory_stats', {})
        if memory_stats.get('avg_percent', 0) > 80:
            recommendations.append("High memory usage detected. Consider increasing batch sizes or implementing memory cleanup.")
        elif memory_stats.get('trend') == 'increasing':
            recommendations.append("Memory usage is trending upward. Monitor for potential memory leaks.")

        # CPU recommendations
        cpu_stats = summary.get('cpu_stats', {})
        if cpu_stats.get('avg_percent', 0) > 70:
            recommendations.append("High CPU usage detected. Consider reducing parallel processing or optimizing algorithms.")
        elif cpu_stats.get('trend') == 'increasing':
            recommendations.append("CPU usage is increasing. Consider load balancing or resource optimization.")

        # Thread recommendations
        process_stats = summary.get('process_stats', {})
        if process_stats.get('avg_threads', 0) > 50:
            recommendations.append("High thread count detected. Consider thread pool optimization.")
        elif process_stats.get('thread_trend') == 'increasing':
            recommendations.append("Thread count is increasing. Monitor for thread leaks.")

        # Disk recommendations
        disk_stats = summary.get('disk_stats', {})
        if disk_stats.get('avg_percent', 0) > 90:
            recommendations.append("High disk usage detected. Consider cleanup or additional storage.")

        return recommendations

    def _check_resource_alerts(self, summary: Dict[str, Any]) -> List[str]:
        """Check for resource alerts based on current usage"""
        alerts = []

        # Memory alerts
        memory_stats = summary.get('memory_stats', {})
        if memory_stats.get('max_percent', 0) > self.thresholds['memory_percent']:
            alerts.append(f"Memory usage exceeded threshold: {memory_stats['max_percent']:.1f}%")

        # CPU alerts
        cpu_stats = summary.get('cpu_stats', {})
        if cpu_stats.get('max_percent', 0) > self.thresholds['cpu_percent']:
            alerts.append(f"CPU usage exceeded threshold: {cpu_stats['max_percent']:.1f}%")

        # Disk alerts
        disk_stats = summary.get('disk_stats', {})
        if disk_stats.get('max_percent', 0) > self.thresholds['disk_percent']:
            alerts.append(f"Disk usage exceeded threshold: {disk_stats['max_percent']:.1f}%")

        return alerts

    def set_threshold(self, resource: str, value: float):
        """Set a resource usage threshold"""
        if resource in self.thresholds:
            self.thresholds[resource] = value
            log_function(f"[RESOURCE] Updated {resource} threshold to {value}", LOG_INFO)

    def export_resource_data(self, filepath: str, format: str = "json") -> bool:
        """Export resource monitoring data"""
        try:
            if format.lower() == "json":
                data = {
                    'export_timestamp': time.time(),
                    'resource_report': self.get_resource_report(),
                    'full_history': [s.to_dict() for s in self.snapshots]
                }

                with open(filepath, 'w') as f:
                    import json
                    json.dump(data, f, indent=2)

            log_function(f"[RESOURCE] Resource data exported to {filepath}", LOG_INFO)
            return True

        except Exception as e:
            log_function(f"[RESOURCE] Failed to export resource data: {str(e)}", LOG_ERROR)
            return False


# Global resource monitor instance
resource_monitor = ResourceMonitor()


def get_resource_monitor() -> ResourceMonitor:
    """Get the global resource monitor instance"""
    return resource_monitor


def get_current_resource_usage() -> Dict[str, Any]:
    """Get current resource usage snapshot"""
    snapshot = resource_monitor.get_current_snapshot()
    return snapshot.to_dict() if snapshot else {}


def log_resource_usage():
    """Log current resource usage"""
    snapshot = resource_monitor.get_current_snapshot()
    if snapshot:
        log_function("[RESOURCE] === Current Resource Usage ===", LOG_INFO)

        if snapshot.memory_info:
            log_function(f"[RESOURCE] Memory: {snapshot.memory_info.get('percent', 0):.1f}% "
                        f"({snapshot.memory_info.get('used_mb', 0):.0f}MB used)", LOG_INFO)

        if snapshot.cpu_info:
            log_function(f"[RESOURCE] CPU: {snapshot.cpu_info.get('percent', 0):.1f}%", LOG_INFO)

        if snapshot.process_info:
            log_function(f"[RESOURCE] Process: {snapshot.process_info.get('threads', 0)} threads, "
                        f"{snapshot.process_info.get('memory_rss_mb', 0):.0f}MB memory", LOG_INFO)

        if snapshot.disk_info:
            log_function(f"[RESOURCE] Disk: {snapshot.disk_info.get('usage_percent', 0):.1f}% used", LOG_INFO)


# Convenience functions for quick resource checks
def get_memory_usage_percent() -> float:
    """Get current memory usage percentage"""
    snapshot = resource_monitor.get_current_snapshot()
    return snapshot.memory_info.get('percent', 0) if snapshot and snapshot.memory_info else 0


def get_cpu_usage_percent() -> float:
    """Get current CPU usage percentage"""
    snapshot = resource_monitor.get_current_snapshot()
    return snapshot.cpu_info.get('percent', 0) if snapshot and snapshot.cpu_info else 0


def get_thread_count() -> int:
    """Get current thread count"""
    snapshot = resource_monitor.get_current_snapshot()
    return snapshot.process_info.get('threads', 0) if snapshot and snapshot.process_info else 0


def is_resource_usage_high() -> bool:
    """Check if any resource usage is above normal thresholds"""
    snapshot = resource_monitor.get_current_snapshot()
    if not snapshot:
        return False

    memory_high = snapshot.memory_info.get('percent', 0) > 80 if snapshot.memory_info else False
    cpu_high = snapshot.cpu_info.get('percent', 0) > 80 if snapshot.cpu_info else False
    disk_high = snapshot.disk_info.get('usage_percent', 0) > 90 if snapshot.disk_info else False

    return memory_high or cpu_high or disk_high