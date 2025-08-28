# Sub-AI Translator Performance Optimization Plan

## Current Performance Issues Analysis

Based on code analysis and research, the following performance bottlenecks have been identified:

1. **Connection Management**: The application uses `urllib.request` without connection pooling, creating a new TCP connection for each API request, which introduces significant overhead.

2. **Conservative Rate Limiting**: The backoff mechanism has overly conservative settings, particularly for OpenRouter (1 second minimum interval) which significantly slows processing.

3. **Limited Parallel Processing**:

   - Gemini is hardcoded to parallel=1
   - OpenRouter is capped at max 3 parallel requests
   - Current batch sizes are quite small (10-20 items per batch)

4. **Suboptimal Timeout Settings**:

   - OpenRouter and Gemini have 10-second timeouts
   - No distinction between connection and read timeouts

5. **Inefficient Retry Logic**: Multiple retry attempts with exponential backoff can significantly increase processing time when errors occur.

## Proposed Optimizations

### 1. Connection Pooling Implementation

**Issue**: Current implementation creates a new TCP connection for each API request.

**Solution**: Implement connection pooling using `urllib3` or `requests` library:

- Use `urllib3.PoolManager` with appropriate pool size settings
- Reuse connections for multiple requests to the same host
- Configure pool size based on parallel processing settings

**Implementation**:

```python
# Example implementation approach
import urllib3

# Create a pool manager with appropriate settings
http = urllib3.PoolManager(
    num_pools=4,  # Number of connection pools
    maxsize=10,   # Max connections per pool
    retries=False, # We handle retries ourselves
    timeout=urllib3.Timeout(connect=5.0, read=30.0)
)
```

### 2. Optimized Rate Limiting Configuration

**Issue**: Current rate limiting is overly conservative.

**Solution**: Adjust rate limiting parameters based on provider capabilities:

- Increase minimum intervals for providers that support higher rate limits
- Implement smarter backoff strategies that consider actual rate limit responses
- Use sliding window rate limiting instead of fixed intervals

**Implementation**:

```python
# Example configuration adjustments
PROVIDER_CONFIG = {
    "OpenAI": {
        "min_interval": 0.05,  # Much more aggressive for OpenAI
        "retries": 2,
        "base_delay": 0.5,
        "max_delay": 5.0
    },
    "Gemini": {
        "min_interval": 0.1,   # More aggressive for Gemini
        "retries": 2,
        "base_delay": 1.0,
        "max_delay": 8.0
    },
    "OpenRouter": {
        "min_interval": 0.2,   # Much more aggressive
        "retries": 3,
        "base_delay": 1.0,
        "max_delay": 15.0
    }
}
```

### 3. Enhanced Parallel Processing

**Issue**: Limited parallel processing capabilities.

**Solution**:

- Increase parallel processing limits for all providers
- Implement dynamic parallel processing based on provider response times
- Allow user configuration of parallel processing limits

**Implementation**:

```python
# Updated provider configurations
PROVIDER_BATCH_CONFIG = {
    "OpenAI": {
        "max_batch_size": 50,      # Increase batch size
        "max_content_length": 15000,
        "max_parallel": 10         # Increase parallel requests
    },
    "Gemini": {
        "max_batch_size": 30,
        "max_content_length": 20000,
        "max_parallel": 5          # Enable parallel processing
    },
    "OpenRouter": {
        "max_batch_size": 25,
        "max_content_length": 12000,
        "max_parallel": 8          # Increase parallel requests
    }
}
```

### 4. Improved Timeout Management

**Issue**: Uniform timeout settings don't account for different request types.

**Solution**:

- Implement separate connect and read timeouts
- Use adaptive timeouts based on request size
- Add timeout configuration options for users

**Implementation**:

```python
# Example timeout configuration
DEFAULT_TIMEOUTS = {
    "connect": 5.0,   # Connection timeout
    "read": 30.0,     # Read timeout
    "total": 60.0     # Total request timeout
}
```

### 5. Optimized Batch Processing

**Issue**: Fixed batch sizes don't account for content complexity.

**Solution**:

- Implement dynamic batch sizing based on content complexity
- Use character count and line count for better batch sizing
- Add batch size configuration options

**Implementation**:

```python
def calculate_dynamic_batch_size(batch, max_content_length, max_lines=100):
    """Calculate appropriate batch size based on content complexity"""
    if not batch:
        return 1

    # Calculate total content metrics
    total_chars = sum(len("\n".join(b["lines"])) for _, b in batch)
    total_lines = sum(len(b["lines"]) for _, b in batch)

    # Adjust batch size based on both character count and line count
    if total_chars < max_content_length // 3 and total_lines < max_lines // 2:
        return min(len(batch), max(1, int(max_content_length / (total_chars / len(batch) if len(batch) > 0 else 1))))

    # Reduce batch size to fit within limits
    current_size = len(batch)
    while current_size > 0:
        estimated_chars = (total_chars / len(batch)) * current_size
        estimated_lines = (total_lines / len(batch)) * current_size
        if estimated_chars <= max_content_length and estimated_lines <= max_lines:
            return current_size
        current_size -= 1

    return 1
```

## Implementation Roadmap

### Phase 1: Connection Pooling and Basic Optimizations

- Implement connection pooling using urllib3
- Update API modules to use connection pooling
- Adjust timeout settings

### Phase 2: Rate Limiting and Parallel Processing

- Optimize rate limiting configurations
- Increase parallel processing limits
- Implement dynamic parallel processing

### Phase 3: Batch Processing and Content Optimization

- Implement dynamic batch sizing
- Optimize content length calculations
- Add user configuration options

### Phase 4: Monitoring and Testing

- Add performance monitoring and logging
- Implement comprehensive testing
- Optimize based on test results

## Expected Performance Improvements

1. **Connection Overhead Reduction**: 50-80% reduction in connection establishment time
2. **Increased Throughput**: 2-5x increase in translation speed through better parallelization
3. **Reduced Latency**: 30-60% reduction in average request latency
4. **Better Error Handling**: More efficient retry mechanisms reducing total processing time

## Risk Mitigation

1. **Rate Limit Compliance**: Ensure optimizations don't violate provider rate limits
2. **Backward Compatibility**: Maintain compatibility with existing configurations
3. **Error Handling**: Preserve robust error handling while improving performance
4. **Testing**: Comprehensive testing to ensure optimizations don't introduce bugs

## Monitoring and Metrics

- Request duration tracking
- Connection pool utilization metrics
- Error rate monitoring
- Throughput measurements
- User-configurable logging levels
