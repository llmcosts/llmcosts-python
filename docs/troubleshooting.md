# Troubleshooting Guide

This guide covers common issues and solutions when using LLMCosts.

## 🔧 Common Issues

### 1. Missing API Key Error

**Error:**
```
ValueError: LLMCOSTS_API_KEY is required
```

**Solution:**
Set the `LLMCOSTS_API_KEY` environment variable or pass `api_key` parameter:

```python
# Method 1: Environment variable
export LLMCOSTS_API_KEY="your-api-key"

# Method 2: Direct parameter
tracked_client = LLMTrackingProxy(
    client, 
    provider=Provider.OPENAI,
    api_key="your-llmcosts-api-key"
)
```

### 2. OpenAI Streaming Without Usage

**Error:**
```
stream_options={"include_usage": True} required for OpenAI streaming
```

**Solution:**
Add `stream_options={"include_usage": True}` to OpenAI streaming calls:

```python
# ✅ Correct streaming setup
stream = tracked_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Count to 10"}],
    stream=True,
    stream_options={"include_usage": True}  # Required for usage tracking
)
```

### 3. LangChain Integration Issues

**Error: `'generator' object does not support the context manager protocol`**

```python
# ❌ Wrong - don't pass the full tracked client
chat_model = ChatOpenAI(client=tracked_client)  # Wrong!

# ✅ Correct - use the appropriate sub-client
chat_model = ChatOpenAI(client=tracked_client.chat.completions)  # Correct!
```

**Error: `AttributeError: 'OpenAI' object has no attribute 'create'`**

```python
# ❌ Wrong - don't pass the full tracked client
llm = OpenAI(client=tracked_client)  # Wrong!

# ✅ Correct - use the completions sub-client
llm = OpenAI(client=tracked_client.completions)  # Correct!
```

### 4. Tracker Not Starting

**Check tracker health:**

```python
from llmcosts import get_usage_tracker

tracker = get_usage_tracker()
health = tracker.get_health_info()
print(health)

# Expected output:
# {
#   'status': 'running',
#   'total_sent': 10,
#   'queue_size': 0,
#   'last_error': None
# }
```

**Common solutions:**
- Verify your API key is correct
- Check network connectivity
- Ensure the endpoint is reachable

### 5. Queue Full Warnings

**Warning:**
```
Usage queue is full. Dropping usage data.
```

**Solution:**
Increase `max_queue_size` or check network connectivity:

```python
# Check queue status
tracker = get_usage_tracker()
health = tracker.get_health_info()
print(f"Queue size: {health['queue_size']}")

# If consistently full, there may be network issues
# or the endpoint might be unreachable
```

### 6. No Usage Logs Appearing

**Debug steps:**

```python
# Enable debug mode to see what's happening
tracked_client = LLMTrackingProxy(
    client,
    provider=Provider.OPENAI,
    debug=True,
    sync_mode=True  # Wait for tracking completion
)

# Make a test call
response = tracked_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Test"}]
)

# Look for logs like: "[LLM costs] OpenAI usage →"
```

**Common causes:**
- Missing or incorrect API key
- Network connectivity issues
- Incorrect provider specification
- Framework parameter needed but not set

## 🔍 Debugging Techniques

### Enable Debug Logging

```python
import logging

# Enable debug logging for LLMCosts
logging.basicConfig(level=logging.DEBUG)

# Create tracked client with debug mode
tracked_client = LLMTrackingProxy(
    client,
    provider=Provider.OPENAI,
    debug=True,
    sync_mode=True
)
```

### Check Tracker Status

```python
from llmcosts import get_usage_tracker

tracker = get_usage_tracker()

# Get health information
health = tracker.get_health_info()
print(f"Status: {health['status']}")
print(f"Queue size: {health['queue_size']}")
print(f"Total sent: {health['total_sent']}")
print(f"Last error: {health.get('last_error')}")

# Get last response (for debugging)
last_response = tracker.get_last_response()
if last_response:
    print(f"Last response: {last_response}")
```

### Test with Sync Mode

```python
# Use sync mode for testing to ensure tracking completes
tracked_client = LLMTrackingProxy(
    client,
    provider=Provider.OPENAI,
    sync_mode=True,  # Wait for completion
    debug=True,      # Enable debug output
    remote_save=False  # Don't save to remote during testing
)

# Make a call and verify tracking
response = tracked_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Test"}]
)
# Tracking will complete before this point
```

## 🌐 Network and Endpoint Issues

### Custom Endpoint Configuration

If you're using a custom endpoint:

```python
import os
from llmcosts.tracker import reset_global_tracker

# Set custom endpoint
os.environ["LLMCOSTS_API_ENDPOINT"] = "https://your-endpoint.com/api/v1/usage"

# Force tracker recreation
reset_global_tracker()

# Now create your tracked client
tracked_client = LLMTrackingProxy(client, provider=Provider.OPENAI)
```

### Verify Endpoint Connectivity

```python
import requests
import os

# Test endpoint connectivity
endpoint = os.environ.get("LLMCOSTS_API_ENDPOINT", "https://api.llmcosts.com/api/v1/usage")
api_key = os.environ.get("LLMCOSTS_API_KEY")

try:
    response = requests.get(
        endpoint.replace("/usage", "/health"),  # Health check endpoint
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10
    )
    print(f"Endpoint status: {response.status_code}")
except requests.exceptions.RequestException as e:
    print(f"Endpoint unreachable: {e}")
```

## 📊 Response Callback Issues

### SQLite Callback Not Working

**Check the target path:**

```python
import os

# Verify the environment variable is set
path = os.environ.get('SQLITE_CALLBACK_TARGET_PATH')
print(f"SQLite target path: {path}")

# Check if directory exists
if path and not os.path.exists(path):
    os.makedirs(path)
    print(f"Created directory: {path}")
```

### Text File Callback Not Working

**Check the target path:**

```python
import os

# Verify the environment variable is set
path = os.environ.get('TEXT_CALLBACK_TARGET_PATH')
print(f"Text file target path: {path}")

# Check if directory exists
if path and not os.path.exists(path):
    os.makedirs(path)
    print(f"Created directory: {path}")
```

## 🧪 Testing and Validation

### Unit Test Setup

```python
import unittest
from llmcosts.tracker import LLMTrackingProxy, Provider

class TestLLMCosts(unittest.TestCase):
    def setUp(self):
        self.tracked_client = LLMTrackingProxy(
            mock_client,
            provider=Provider.OPENAI,
            sync_mode=True,     # Wait for completion
            debug=True,         # Enable debug output
            remote_save=False   # Don't save during testing
        )
    
    def test_usage_tracking(self):
        # Your test code here
        pass
```

### Integration Test

```python
from llmcosts import LLMTrackingProxy, Provider
import openai

def test_integration():
    """Test actual API integration."""
    client = openai.OpenAI(api_key="your-test-key")
    tracked_client = LLMTrackingProxy(
        client,
        provider=Provider.OPENAI,
        debug=True,
        sync_mode=True
    )
    
    # Make a small test call
    response = tracked_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=5
    )
    
    print(f"Response: {response.choices[0].message.content}")
    print("✅ Integration test passed")

if __name__ == "__main__":
    test_integration()
```

## 📞 Getting Help

If you're still experiencing issues:

1. **Check the logs** - Enable debug mode and look for error messages
2. **Verify your setup** - Ensure all required parameters are provided
3. **Test with sync mode** - Use `sync_mode=True` to see if tracking works
4. **Check network connectivity** - Ensure the endpoint is reachable
5. **Contact support** - Email [help@llmcosts.com](mailto:help@llmcosts.com) with:
   - Your configuration code
   - Error messages
   - Debug logs
   - Expected vs actual behavior

## 🔗 Related Documentation

- **[Configuration](configuration.md)** - Advanced configuration options
- **[Providers](providers.md)** - Provider-specific integration guides
- **[Testing](testing.md)** - Comprehensive testing documentation 