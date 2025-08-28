"""
Cost Analysis and Optimization System

This module provides comprehensive cost tracking, analysis, and optimization
insights for API usage across different providers.

Features:
- Real-time cost tracking and estimation
- Provider cost comparison and optimization
- Budget monitoring and alerts
- Cost-benefit analysis for different strategies
- Usage pattern analysis for cost optimization
- Cost forecasting and planning
- Automated cost optimization recommendations

Expected benefits:
- Cost transparency and control
- Data-driven provider selection
- Budget optimization
- Cost reduction through intelligent usage patterns
- Proactive cost management
"""

import time
import statistics
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import json

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
    log_function = lambda msg, level=0: print(f"[COST] {msg}")

# Import performance monitoring
try:
    from .performance_monitor import get_performance_monitor, record_cost_metrics
    PERFORMANCE_MONITORING_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITORING_AVAILABLE = False


class ProviderCostProfile:
    """Cost profile for a specific API provider"""

    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        self.base_costs = {}  # Model-specific base costs
        self.usage_costs = {}  # Model-specific usage costs
        self.free_tier_limits = {}  # Free tier limits if any
        self.billing_period = "monthly"  # monthly, daily, etc.
        self.currency = "USD"

        # Initialize with known provider costs
        self._initialize_provider_costs()

    def _initialize_provider_costs(self):
        """Initialize cost profiles for known providers"""
        if self.provider_name.lower() == "openai":
            self.base_costs = {
                "gpt-4o-mini": {"input": 0.15, "output": 0.60},  # per 1M tokens
                "gpt-4o": {"input": 2.50, "output": 10.00},
                "gpt-4-turbo": {"input": 10.00, "output": 30.00},
                "gpt-4": {"input": 30.00, "output": 60.00},
                "gpt-3.5-turbo": {"input": 0.50, "output": 1.50}
            }
            self.currency = "USD"
            self.billing_period = "monthly"

        elif self.provider_name.lower() == "gemini":
            self.base_costs = {
                "gemini-2.0-flash-exp": {"input": 0.00, "output": 0.00},  # Free during experimental
                "gemini-1.5-pro": {"input": 1.25, "output": 5.00},  # per 1M tokens
                "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
                "gemini-1.0-pro": {"input": 0.50, "output": 1.50}
            }
            self.currency = "USD"
            self.billing_period = "monthly"

        elif self.provider_name.lower() == "openrouter":
            # OpenRouter costs vary by model, using approximate averages
            self.base_costs = {
                "openai/gpt-4o-mini": {"input": 0.20, "output": 0.80},
                "openai/gpt-4o": {"input": 3.00, "output": 12.00},
                "anthropic/claude-3-haiku": {"input": 0.25, "output": 1.25},
                "anthropic/claude-3-sonnet": {"input": 3.00, "output": 15.00},
                "meta-llama/llama-2-70b-chat": {"input": 0.50, "output": 0.80}
            }
            self.currency = "USD"
            self.billing_period = "monthly"

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a specific model and token usage"""
        if model not in self.base_costs:
            # Try to find a similar model or use default
            model = self._find_similar_model(model)
            if not model:
                log_function(f"[COST] Unknown model {model} for provider {self.provider_name}, using default", LOG_WARNING)
                return 0.0

        costs = self.base_costs[model]
        input_cost = (input_tokens / 1_000_000) * costs.get("input", 0)
        output_cost = (output_tokens / 1_000_000) * costs.get("output", 0)

        return input_cost + output_cost

    def _find_similar_model(self, model: str) -> Optional[str]:
        """Find a similar model in the cost profile"""
        model_lower = model.lower()

        # Try exact matches first
        for known_model in self.base_costs.keys():
            if known_model.lower() in model_lower or model_lower in known_model.lower():
                return known_model

        # Try partial matches
        for known_model in self.base_costs.keys():
            if any(part in model_lower for part in known_model.lower().split('-')):
                return known_model

        return None

    def get_cost_per_token(self, model: str) -> Dict[str, float]:
        """Get cost per million tokens for a model"""
        if model in self.base_costs:
            return self.base_costs[model].copy()
        return {"input": 0.0, "output": 0.0}


class CostTracker:
    """Tracks and analyzes API usage costs"""

    def __init__(self):
        self.provider_profiles = {}
        self.usage_history = []
        self.cost_history = []
        self.budgets = {}
        self.alerts = []

        # Initialize provider profiles
        self._initialize_provider_profiles()

    def _initialize_provider_profiles(self):
        """Initialize cost profiles for all supported providers"""
        providers = ["openai", "gemini", "openrouter"]
        for provider in providers:
            self.provider_profiles[provider] = ProviderCostProfile(provider)

    def track_api_call(self, provider: str, model: str, input_tokens: int,
                      output_tokens: int, success: bool = True) -> float:
        """Track an API call and calculate its cost"""
        if provider not in self.provider_profiles:
            log_function(f"[COST] Unknown provider {provider}, skipping cost tracking", LOG_WARNING)
            return 0.0

        profile = self.provider_profiles[provider]
        cost = profile.estimate_cost(model, input_tokens, output_tokens)

        # Record usage
        usage_record = {
            'timestamp': time.time(),
            'provider': provider,
            'model': model,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'cost': cost,
            'success': success
        }

        self.usage_history.append(usage_record)
        self.cost_history.append({
            'timestamp': time.time(),
            'provider': provider,
            'cost': cost
        })

        # Keep only recent history (last 1000 records)
        if len(self.usage_history) > 1000:
            self.usage_history = self.usage_history[-500:]
        if len(self.cost_history) > 1000:
            self.cost_history = self.cost_history[-500:]

        # Record in performance monitor
        if PERFORMANCE_MONITORING_AVAILABLE:
            record_cost_metrics(cost, input_tokens + output_tokens, provider)

        return cost

    def get_cost_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get cost summary for the specified time window"""
        cutoff_time = time.time() - (time_window_hours * 3600)

        # Filter records within time window
        recent_usage = [u for u in self.usage_history if u['timestamp'] > cutoff_time]
        recent_costs = [c for c in self.cost_history if c['timestamp'] > cutoff_time]

        if not recent_usage:
            return {
                'time_window_hours': time_window_hours,
                'total_cost': 0.0,
                'total_tokens': 0,
                'api_calls': 0,
                'avg_cost_per_call': 0.0,
                'cost_per_token': 0.0,
                'provider_breakdown': {},
                'model_breakdown': {}
            }

        # Calculate totals
        total_cost = sum(u['cost'] for u in recent_usage)
        total_tokens = sum(u['total_tokens'] for u in recent_usage)
        total_calls = len(recent_usage)

        # Provider breakdown
        provider_breakdown = defaultdict(float)
        for usage in recent_usage:
            provider_breakdown[usage['provider']] += usage['cost']

        # Model breakdown
        model_breakdown = defaultdict(float)
        for usage in recent_usage:
            model_key = f"{usage['provider']}/{usage['model']}"
            model_breakdown[model_key] += usage['cost']

        # Calculate averages
        avg_cost_per_call = total_cost / total_calls if total_calls > 0 else 0
        cost_per_token = total_cost / total_tokens if total_tokens > 0 else 0

        return {
            'time_window_hours': time_window_hours,
            'total_cost': total_cost,
            'total_tokens': total_tokens,
            'api_calls': total_calls,
            'avg_cost_per_call': avg_cost_per_call,
            'cost_per_token': cost_per_token,
            'provider_breakdown': dict(provider_breakdown),
            'model_breakdown': dict(model_breakdown),
            'cost_trend': self._calculate_cost_trend(recent_costs)
        }

    def _calculate_cost_trend(self, costs: List[Dict]) -> str:
        """Calculate cost trend direction"""
        if len(costs) < 6:
            return 'stable'

        # Get costs from first and second half
        mid_point = len(costs) // 2
        first_half = [c['cost'] for c in costs[:mid_point]]
        second_half = [c['cost'] for c in costs[mid_point:]]

        if first_half and second_half:
            first_avg = statistics.mean(first_half)
            second_avg = statistics.mean(second_half)

            if second_avg > first_avg * 1.1:
                return 'increasing'
            elif second_avg < first_avg * 0.9:
                return 'decreasing'

        return 'stable'

    def set_budget(self, provider: str, amount: float, period: str = "monthly"):
        """Set budget limit for a provider"""
        self.budgets[provider] = {
            'amount': amount,
            'period': period,
            'start_time': time.time()
        }
        log_function(f"[COST] Set {period} budget for {provider}: {amount}", LOG_INFO)

    def check_budget_alerts(self) -> List[str]:
        """Check for budget alerts"""
        alerts = []

        for provider, budget in self.budgets.items():
            current_cost = self.get_provider_cost(provider, budget['period'])

            if current_cost >= budget['amount'] * 0.8:  # 80% threshold
                alerts.append(f"Budget alert: {provider} has used {current_cost:.2f} of {budget['amount']:.2f} {budget['period']} budget")

            if current_cost >= budget['amount']:
                alerts.append(f"Budget exceeded: {provider} has exceeded {budget['period']} budget ({current_cost:.2f} > {budget['amount']:.2f})")

        return alerts

    def get_provider_cost(self, provider: str, period: str = "monthly") -> float:
        """Get total cost for a provider in the specified period"""
        period_seconds = self._period_to_seconds(period)
        cutoff_time = time.time() - period_seconds

        provider_costs = [c for c in self.cost_history
                         if c['timestamp'] > cutoff_time and c['provider'] == provider]

        return sum(c['cost'] for c in provider_costs)

    def _period_to_seconds(self, period: str) -> int:
        """Convert period string to seconds"""
        period_map = {
            'hourly': 3600,
            'daily': 86400,
            'weekly': 604800,
            'monthly': 2592000,  # 30 days
            'yearly': 31536000
        }
        return period_map.get(period.lower(), 2592000)  # Default to monthly

    def get_cost_forecast(self, provider: str, days_ahead: int = 30) -> Dict[str, Any]:
        """Forecast future costs based on usage patterns"""
        # Get historical data for the provider
        provider_usage = [u for u in self.usage_history if u['provider'] == provider]

        if len(provider_usage) < 7:  # Need at least a week of data
            return {
                'forecast_period_days': days_ahead,
                'forecasted_cost': 0.0,
                'confidence': 'low',
                'based_on_days': len(set(int(u['timestamp'] / 86400) for u in provider_usage))
            }

        # Calculate daily averages
        daily_costs = defaultdict(float)
        for usage in provider_usage:
            day = int(usage['timestamp'] / 86400)
            daily_costs[day] += usage['cost']

        daily_avg_cost = statistics.mean(daily_costs.values())
        daily_std_cost = statistics.stdev(daily_costs.values()) if len(daily_costs) > 1 else 0

        forecasted_cost = daily_avg_cost * days_ahead

        # Calculate confidence based on data consistency
        cv = daily_std_cost / daily_avg_cost if daily_avg_cost > 0 else 0
        if cv < 0.2:
            confidence = 'high'
        elif cv < 0.5:
            confidence = 'medium'
        else:
            confidence = 'low'

        return {
            'forecast_period_days': days_ahead,
            'forecasted_cost': forecasted_cost,
            'daily_average': daily_avg_cost,
            'confidence': confidence,
            'based_on_days': len(daily_costs),
            'coefficient_of_variation': cv
        }

    def get_optimization_recommendations(self) -> List[str]:
        """Generate cost optimization recommendations"""
        recommendations = []

        summary = self.get_cost_summary(168)  # Last 7 days

        if not summary['api_calls']:
            return ["Insufficient data for cost optimization recommendations"]

        # Analyze provider costs
        provider_breakdown = summary['provider_breakdown']
        if len(provider_breakdown) > 1:
            # Find most expensive provider
            most_expensive = max(provider_breakdown.items(), key=lambda x: x[1])
            cheapest = min(provider_breakdown.items(), key=lambda x: x[1])

            if most_expensive[1] > cheapest[1] * 1.5:
                recommendations.append(f"Consider switching from {most_expensive[0]} to {cheapest[0]} - potential savings of ${(most_expensive[1] - cheapest[1]):.2f}")

        # Analyze model usage
        model_breakdown = summary['model_breakdown']
        if len(model_breakdown) > 1:
            most_expensive_model = max(model_breakdown.items(), key=lambda x: x[1])
            recommendations.append(f"Your most expensive model is {most_expensive_model[0]} at ${most_expensive_model[1]:.2f}")

        # Cost per token analysis
        cost_per_token = summary['cost_per_token']
        if cost_per_token > 0.01:  # More than $0.01 per 1M tokens
            recommendations.append(f"High cost per token (${cost_per_token:.4f}). Consider using cheaper models or optimizing prompts")

        # Usage pattern analysis
        if summary['cost_trend'] == 'increasing':
            recommendations.append("Costs are trending upward. Review recent usage patterns and consider optimization")

        # Budget alerts
        budget_alerts = self.check_budget_alerts()
        recommendations.extend(budget_alerts)

        return recommendations

    def compare_providers(self, models_to_compare: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Compare costs between different provider/model combinations"""
        comparison = {}

        for provider_model in models_to_compare:
            provider, model = provider_model.split('/', 1)

            if provider in self.provider_profiles:
                profile = self.provider_profiles[provider]
                costs = profile.get_cost_per_token(model)
                comparison[provider_model] = costs

        return comparison

    def export_cost_data(self, filepath: str, format: str = "json") -> bool:
        """Export cost data for external analysis"""
        try:
            if format.lower() == "json":
                data = {
                    'export_timestamp': time.time(),
                    'cost_summary': self.get_cost_summary(),
                    'usage_history': self.usage_history[-500:],  # Last 500 records
                    'budgets': self.budgets,
                    'optimization_recommendations': self.get_optimization_recommendations()
                }

                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)

            log_function(f"[COST] Cost data exported to {filepath}", LOG_INFO)
            return True

        except Exception as e:
            log_function(f"[COST] Failed to export cost data: {str(e)}", LOG_ERROR)
            return False


# Global cost tracker instance
cost_tracker = CostTracker()


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance"""
    return cost_tracker


def estimate_translation_cost(provider: str, model: str, text_length: int,
                           estimated_output_multiplier: float = 1.2) -> float:
    """Estimate cost for translating text of given length"""
    # Rough token estimation: ~4 characters per token for input, output multiplier for response
    estimated_input_tokens = text_length // 4
    estimated_output_tokens = int(estimated_input_tokens * estimated_output_multiplier)

    return cost_tracker.track_api_call(provider, model, estimated_input_tokens,
                                     estimated_output_tokens, success=True)


def log_cost_summary():
    """Log current cost summary"""
    summary = cost_tracker.get_cost_summary()

    log_function("[COST] === Cost Summary (24h) ===", LOG_INFO)
    log_function(f"[COST] Total cost: ${summary['total_cost']:.2f}", LOG_INFO)
    log_function(f"[COST] API calls: {summary['api_calls']}", LOG_INFO)
    log_function(f"[COST] Tokens used: {summary['total_tokens']:,}", LOG_INFO)
    log_function(f"[COST] Avg cost per call: ${summary['avg_cost_per_call']:.4f}", LOG_INFO)

    if summary['provider_breakdown']:
        log_function("[COST] Provider breakdown:", LOG_INFO)
        for provider, cost in summary['provider_breakdown'].items():
            log_function(f"[COST]   {provider}: ${cost:.2f}", LOG_INFO)


# Convenience functions for common cost operations
def track_openai_cost(model: str, input_tokens: int, output_tokens: int, success: bool = True) -> float:
    """Track OpenAI API cost"""
    return cost_tracker.track_api_call('openai', model, input_tokens, output_tokens, success)


def track_gemini_cost(model: str, input_tokens: int, output_tokens: int, success: bool = True) -> float:
    """Track Gemini API cost"""
    return cost_tracker.track_api_call('gemini', model, input_tokens, output_tokens, success)


def track_openrouter_cost(model: str, input_tokens: int, output_tokens: int, success: bool = True) -> float:
    """Track OpenRouter API cost"""
    return cost_tracker.track_api_call('openrouter', model, input_tokens, output_tokens, success)