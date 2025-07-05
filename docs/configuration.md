# Configuration Guide

This guide covers all configuration options, environment variables, and advanced settings for LLMCosts.

**🔒 Privacy & Security**: LLMCosts is designed with privacy-first principles. We NEVER see your API keys, requests, or responses. Only usage metadata (tokens, costs, model info) is extracted from response objects and transmitted. Your sensitive data stays completely private.

**🔄 Universal Compatibility**: One `LLMTrackingProxy` configuration works with ANY LLM provider's SDK. No need for different configurations per provider.

## 🔧 LLMTrackingProxy Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target` | Any | Required | The LLM client to wrap |
| `provider` | Provider | Required | Provider enum specifying the LLM service (OpenAI, Anthropic, etc.) |
| `framework` | Framework | `None` | **Optional** framework integration (e.g., `Framework.LANGCHAIN`). **Usually `None` for direct API usage.** Only needed for special integrations. |
| `debug` | bool | `False` | Enable debug logging |
| `sync_mode` | bool | `False` | Wait for usage tracker (good for testing) |
| `remote_save` | bool | `True` | Save usage events to remote server |
| `context` | dict | `None` | Custom context data for tracking |
| `response_callback` | callable | `None` | Function to process responses |
| `api_key` | str | `None` | LLMCOSTS API key (uses env var if not provided) |
| `client_customer_key` | str | `None` | Customer key for multi-tenant applications |

## 🌍 Environment Variables

Configure the global tracker with environment variables:

```bash
# Required: Your LLMCosts API key
export LLMCOSTS_API_KEY="your-api-key"

# Optional: Custom API endpoint (only used when creating new trackers)
export LLMCOSTS_API_ENDPOINT="https://your-endpoint.com/api/v1/usage"

# Built-in callback configuration
export SQLITE_CALLBACK_TARGET_PATH="./data"    # SQLite database location
export TEXT_CALLBACK_TARGET_PATH="./logs"      # Text file location
```

When environment variables are set, you can omit the `api_key` parameter:

```python
# Uses LLMCOSTS_API_KEY from environment
tracked_client = LLMTrackingProxy(client, provider=Provider.OPENAI)
```

> **Note:** The `LLMCOSTS_API_ENDPOINT` environment variable is only read when creating a new tracker instance. To use a custom endpoint, you must set the environment variable before creating any `LLMTrackingProxy` instances or call `reset_global_tracker()` to force recreation of the global tracker.

### Custom Endpoint Configuration

```python
from llmcosts.tracker import reset_global_tracker
import os

# Change endpoint and force tracker recreation
os.environ["LLMCOSTS_API_ENDPOINT"] = "https://your-endpoint.com/api/v1/usage"
reset_global_tracker()  # Force new tracker with custom endpoint

# New proxy instances will now use the custom endpoint
tracked_client = LLMTrackingProxy(client, provider=Provider.OPENAI)
```

## 📋 Dynamic Property Updates

All settings can be changed after initialization:

```python
from llmcosts.tracker import LLMTrackingProxy, Provider
import openai

client = openai.OpenAI(api_key="your-key")
proxy = LLMTrackingProxy(client, provider=Provider.OPENAI)

# Update settings dynamically
proxy.remote_save = False  # Don't save to remote server
proxy.context = {"user_id": "123", "session": "abc"}  # Add tracking context
proxy.client_customer_key = "customer_456"  # Set or change customer key
proxy.sync_mode = True  # Switch to synchronous mode
proxy.response_callback = lambda r: print(f"Response: {r.id}")  # Add callback

# Settings are preserved across sub-clients
chat_client = proxy.chat  # Inherits all parent settings
```

## 🎯 Context Tracking

Track user-specific usage and metadata:

```python
# Track user-specific usage
user_context = {
    "user_id": "user_123",
    "session_id": "session_456",
    "app_version": "1.2.3",
    "environment": "production"
}

tracked_client = LLMTrackingProxy(
    client,
    provider=Provider.OPENAI,  # REQUIRED: Specifies the LLM provider
    # framework=None by default for direct OpenAI API usage
    context=user_context
)

# Context is included in all usage data
response = tracked_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)

# Change context mid-session
tracked_client.context = {"user_id": "user_789", "session_id": "session_999"}
```

## 🏢 Customer Key Tracking

For multi-tenant applications, you can track usage per customer using the `client_customer_key` parameter:

```python
tracked_client = LLMTrackingProxy(
    client,
    provider=Provider.OPENAI,
    client_customer_key="customer_acme_corp"  # Track costs per customer
)

# All API calls automatically include this customer key for billing/analytics
response = tracked_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)
```

> **💡 For comprehensive client tracking examples, multi-tenant patterns, and rich context data, see [Client Tracking & Context Data Guide](client-tracking.md)**

## 📤 Response Callbacks

LLMCosts includes built-in response callbacks for common use cases, and supports custom callbacks for specialized needs.

### Built-in Callbacks

**SQLite Callback** - Records data to a SQLite database:

```python
import os
from llmcosts.tracker import LLMTrackingProxy, Provider
from llmcosts.tracker.callbacks import sqlite_callback

# Set environment variable for database location
os.environ['SQLITE_CALLBACK_TARGET_PATH'] = './data'

# Use with any LLM client
tracked_client = LLMTrackingProxy(
    client,
    provider=Provider.OPENAI,
    response_callback=sqlite_callback
)

# Each API call automatically records cost data to ./data/llm_cost_events.db
response = tracked_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

**Text File Callback** - Records data to JSON Lines format:

```python
import os
from llmcosts.tracker.callbacks import text_callback

# Set environment variable for file location
os.environ['TEXT_CALLBACK_TARGET_PATH'] = './logs'

tracked_client = LLMTrackingProxy(
    client,
    provider=Provider.OPENAI,
    response_callback=text_callback
)

# Each API call automatically records cost data to ./logs/llm_cost_events.jsonl
response = tracked_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Custom Callbacks

Use the built-in callbacks as starting points for your own implementations:

```python
def custom_callback(response):
    """Custom response handler."""
    # Extract basic info
    if hasattr(response, 'id'):
        response_id = response.id
    if hasattr(response, 'usage'):
        tokens = response.usage.total_tokens
        print(f"Response {response_id}: {tokens} tokens used")
    
    # Add your custom logic here
    # - Send to analytics service
    # - Update usage quotas
    # - Log to custom database
    # - Send alerts on high usage

tracked_client = LLMTrackingProxy(
    client,
    provider=Provider.OPENAI,
    response_callback=custom_callback
)
```

### Multiple Callbacks

```python
from llmcosts.tracker.callbacks import sqlite_callback, text_callback

def multi_callback(response):
    """Combine multiple callbacks."""
    sqlite_callback(response)  # Store in database
    text_callback(response)    # Store in text file
    
    # Add custom processing
    if hasattr(response, 'usage') and response.usage.total_tokens > 1000:
        print("⚠️  High token usage detected!")

tracked_client = LLMTrackingProxy(
    client,
    provider=Provider.OPENAI,
    response_callback=multi_callback
)
```

## 🧪 Testing and Debugging

```python
# Enable synchronous mode for testing
tracked_client = LLMTrackingProxy(
    client,
    provider=Provider.OPENAI,  # REQUIRED: Specifies the LLM provider
    # framework=None by default for direct API usage
    sync_mode=True,      # Wait for tracking to complete
    debug=True,          # Enable debug logging
    remote_save=False    # Don't save during testing
)

# Perfect for unit tests
response = tracked_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Test message"}]
)
# Usage tracking completes before this line
```

## 📊 Output Format

Usage data is logged as structured JSON. **This is the ONLY data LLMCosts ever sees or transmits** - no API keys, requests, or responses:

```json
{
  "usage": {
    "completion_tokens": 150,
    "prompt_tokens": 50,
    "total_tokens": 200
  },
  "model_id": "gpt-4o-mini",
  "response_id": "chatcmpl-123abc",
  "timestamp": "2024-01-15T10:30:00Z",
  "provider": "openai",
  "service_tier": "default",
  "context": {
    "user_id": "123",
    "session_id": "abc"
  }
}
```

### Field Descriptions

- **usage**: Token/unit counts extracted from response metadata (varies by provider)
- **model_id**: Model identifier used for the request
- **response_id**: Unique response identifier from the provider
- **timestamp**: ISO 8601 timestamp when the response was processed
- **provider**: LLM provider name
- **service_tier**: Service tier when available (e.g., OpenAI's service tier)
- **context**: Custom tracking data you choose to include
- **remote_save**: Only included when `false`

**🔒 Privacy Note**: This metadata is extracted from the response object's usage/metadata fields only. LLMCosts never accesses request content, response content, or API keys.

## 🏗️ Multi-User Applications

```python
from llmcosts.tracker import LLMTrackingProxy, Provider
import openai
import time

class LLMService:
    def __init__(self, api_key):
        self.base_client = openai.OpenAI(api_key=api_key)
        self.tracked_client = LLMTrackingProxy(
            self.base_client,
            provider=Provider.OPENAI,
            remote_save=True
        )
    
    def chat_for_user(self, user_id, session_id, message):
        # Set user-specific context for tracking
        self.tracked_client.context = {
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": time.time()
        }
        
        return self.tracked_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": message}]
        )

# Usage
service = LLMService("your-api-key")
response1 = service.chat_for_user("user123", "session456", "Hello!")
response2 = service.chat_for_user("user789", "session999", "Hi there!")
```

## 🔍 Advanced: Global Tracker Management

**⚠️ Advanced Use Only**: Most users should stick to `LLMTrackingProxy` which handles tracker management automatically. Only use these functions for debugging, health monitoring, or advanced integrations.

```python
from llmcosts import get_usage_tracker

# ✅ PRIMARY PATTERN (recommended for 95% of use cases)
tracked_client = LLMTrackingProxy(client, provider=Provider.OPENAI)

# ✅ ADVANCED/DEBUGGING ONLY (for the remaining 5%)
# Get the global tracker instance for inspection
tracker = get_usage_tracker()

# Check tracker health
health = tracker.get_health_info()
print(f"Status: {health['status']}")
print(f"Total sent: {health['total_sent']}")
print(f"Queue size: {health['queue_size']}")

# Get last response (for sync mode debugging)
last_response = tracker.get_last_response()
if last_response:
    print(f"Processed: {last_response.get('processed', 0)} records")
```

**Key Points:**
- `LLMTrackingProxy` automatically creates and manages the global tracker
- Child proxies (e.g., `proxy.chat`) reuse the same global tracker  
- Only call `get_usage_tracker()` for debugging or health monitoring
- The global tracker persists across multiple proxy instances

## Next Steps

- **[Providers](providers.md)** - Provider-specific integration guides
- **[Pricing](pricing.md)** - Model discovery and cost calculation
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions 