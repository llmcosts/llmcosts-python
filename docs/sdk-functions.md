# SDK Helper Functions Reference

This document provides comprehensive documentation for all the LLMCosts SDK helper functions that interact with the LLMCosts API. All functions require authentication via API key.

## 📋 Table of Contents

- [Authentication](#authentication)
- [Models & Pricing](#models--pricing)
- [Events & Analytics](#events--analytics)
- [Thresholds & Limits](#thresholds--limits)
- [Health & Diagnostics](#health--diagnostics)
- [Cost Calculation](#cost-calculation)
- [Token Mappings](#token-mappings)

---

## 🔑 Authentication

All functions require an LLMCosts API key. You can provide it either:

```python
# Via parameter
result = function_name(api_key="your-api-key")

# Via environment variable
export LLMCOSTS_API_KEY="your-api-key"
result = function_name()  # Will use env var automatically
```

Get your API key by signing up at [llmcosts.com](https://llmcosts.com).

---

## 🤖 Models & Pricing

### Model Discovery

#### `list_models()`
Get all available models from all providers.

```python
from llmcosts import list_models

models = list_models()
print(f"Total models: {len(models)}")

# Each model contains:
# - provider: LLM provider name
# - model_id: Model identifier
# - aliases: List of known aliases
# - costs: List of token pricing information
```

#### `get_models_dict()`
Get models organized by provider.

```python
from llmcosts import get_models_dict

models_dict = get_models_dict()
print(f"OpenAI models: {len(models_dict.get('openai', []))}")
```

#### `get_models_by_provider(provider)`
Get all model IDs for a specific provider.

```python
from llmcosts import get_models_by_provider, Provider

# Using Provider enum
openai_models = get_models_by_provider(Provider.OPENAI)

# Using string
anthropic_models = get_models_by_provider('anthropic')
```

#### `get_providers_by_model(model_id)`
Find all providers that support a specific model.

```python
from llmcosts import get_providers_by_model

providers = get_providers_by_model('gpt-4')
print(f"GPT-4 is supported by: {', '.join(providers)}")
```

#### `is_model_supported(provider, model_id)`
Check if a provider/model combination is supported.

```python
from llmcosts import is_model_supported, Provider

if is_model_supported(Provider.OPENAI, 'gpt-4'):
    print("OpenAI supports GPT-4")
```

### Pricing Information

#### `get_model_pricing(provider, model_id)`
Get detailed pricing for a specific model.

```python
from llmcosts import get_model_pricing, Provider

pricing = get_model_pricing(Provider.OPENAI, "gpt-4o-mini")
if pricing:
    for cost in pricing['costs']:
        print(f"{cost['token_type']}: ${cost['cost_per_million']}/M tokens")
```

#### `get_token_mappings(provider=None, include_examples=False)`
Get token name mappings for normalization.

```python
from llmcosts import get_token_mappings, Provider

# All providers
mappings = get_token_mappings()

# Specific provider with examples
openai_mappings = get_token_mappings(Provider.OPENAI, include_examples=True)
```

#### `get_provider_token_mappings(provider)`
Get token mappings for a specific provider with examples.

```python
from llmcosts import get_provider_token_mappings, Provider

mappings = get_provider_token_mappings(Provider.OPENAI)
for example in mappings['examples']:
    print(f"Raw: {example['raw_usage']}")
    print(f"Normalized: {example['normalized_tokens']}")
```

---

## 📊 Events & Analytics

### Event Management

#### `list_events(client, **filters)`
List cost events with optional filtering.

```python
from llmcosts import LLMCostsClient, list_events

client = LLMCostsClient()
events = list_events(
    client,
    start="2024-01-01",
    end="2024-01-31",
    provider="openai",
    model_id="gpt-4",
    min_cost=0.01,
    max_cost=1.0
)
```

#### `get_event(client, response_id)`
Get a specific cost event by response ID.

```python
from llmcosts import LLMCostsClient, get_event

client = LLMCostsClient()
event = get_event(client, "your-response-id")
```

#### `search_events(client, **filters)`
Advanced search with aggregations grouped by provider and model.

```python
from llmcosts import LLMCostsClient, search_events

client = LLMCostsClient()
aggregates = search_events(
    client,
    start="2024-01-01",
    provider="openai"
)
```

#### `export_events(client, format="csv", **filters)`
Export cost events as CSV or JSON.

```python
from llmcosts import LLMCostsClient, export_events

client = LLMCostsClient()
csv_data = export_events(
    client,
    format="csv",
    start="2024-01-01",
    end="2024-01-31"
)
```

### Analytics Functions

#### Daily & Monthly Costs

```python
from llmcosts import get_daily_costs, get_monthly_costs

# Daily costs
daily_costs = get_daily_costs(
    start="2024-01-01",
    end="2024-01-31"
)

# Monthly costs
monthly_costs = get_monthly_costs(year=2024)
```

#### Trends & Patterns

```python
from llmcosts import get_cost_trends, get_peak_usage, get_usage_patterns

# Cost trends
trends = get_cost_trends(period="7d")  # 24h, 7d, mtd, ytd

# Peak usage identification
peak = get_peak_usage(days=30)

# Usage patterns by hour/day
patterns = get_usage_patterns()
```

#### Model & Provider Analysis

```python
from llmcosts import (
    get_model_ranking,
    get_model_efficiency,
    get_provider_comparison
)

# Models ranked by cost
model_ranking = get_model_ranking()

# Model efficiency (cost per token)
efficiency = get_model_efficiency()

# Provider comparison
provider_comparison = get_provider_comparison()
```

#### Usage Analysis

```python
from llmcosts import get_usage_frequency, get_usage_outliers

# Usage frequency by day
frequency = get_usage_frequency()

# Identify usage spikes
outliers = get_usage_outliers()
```

---

## 🚨 Thresholds & Limits

### Threshold Management

#### `list_thresholds(client, **filters)`
List all thresholds with optional filtering.

```python
from llmcosts import LLMCostsClient, list_thresholds

client = LLMCostsClient()
thresholds = list_thresholds(
    client,
    type="alert",  # or "limit"
    client_customer_key="customer_123",
    provider="openai",
    model_id="gpt-4"
)
```

#### `create_threshold(client, data)`
Create a new usage threshold.

```python
from llmcosts import LLMCostsClient, create_threshold

client = LLMCostsClient()
threshold = create_threshold(client, {
    "threshold_type": "alert",  # or "limit"
    "amount": 100.0,
    "period": "month",  # or "day"
    "provider": "openai",
    "model_id": "gpt-4",
    "client_customer_key": "customer_123",
    "notification_list": ["email", "webhook"],
    "active": True
})
```

#### `update_threshold(client, threshold_id, data)`
Update an existing threshold.

```python
from llmcosts import LLMCostsClient, update_threshold

client = LLMCostsClient()
updated = update_threshold(client, threshold_id, {
    "amount": 150.0,
    "active": False
})
```

#### `delete_threshold(client, threshold_id)`
Delete a threshold.

```python
from llmcosts import LLMCostsClient, delete_threshold

client = LLMCostsClient()
result = delete_threshold(client, threshold_id)
```

### Threshold Events

#### `list_threshold_events(client, **filters)`
List active threshold events (triggered thresholds).

```python
from llmcosts import LLMCostsClient, list_threshold_events

client = LLMCostsClient()
events = list_threshold_events(
    client,
    type="limit",
    client_customer_key="customer_123"
)
```

---

## 🏥 Health & Diagnostics

#### `health_check()`
Check API health and get triggered threshold info.

```python
from llmcosts import health_check

health = health_check()
print(f"Status: {health['status']}")
print(f"Version: {health['version']}")

# Always includes encrypted triggered threshold info for security
if health.get('triggered_thresholds'):
    print("Triggered thresholds data available")
```

---

## 💰 Cost Calculation

### From Token Counts

#### `calculate_cost_from_tokens(provider, model_id, **tokens)`
Calculate costs from normalized token counts.

```python
from llmcosts import calculate_cost_from_tokens, Provider

cost_result = calculate_cost_from_tokens(
    Provider.OPENAI,
    "gpt-4o-mini",
    input_tokens=1000,
    output_tokens=500,
    cache_read_tokens=100,
    include_explanation=True
)

print(f"Total cost: ${cost_result['costs']['total_cost']}")
print(f"Input cost: ${cost_result['costs']['input_cost']}")
print(f"Output cost: ${cost_result['costs']['output_cost']}")
```

### From Raw Usage

#### `calculate_cost_from_usage(provider, model_id, usage, include_explanation=False)`
Calculate costs from provider-specific usage data.

```python
from llmcosts import calculate_cost_from_usage, Provider

# From OpenAI response
usage_data = {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150
}

cost_result = calculate_cost_from_usage(
    Provider.OPENAI,
    "gpt-4o-mini",
    usage_data,
    include_explanation=True
)

# Detailed explanations available
for explanation in cost_result.get('explanations', []):
    print(f"{explanation['token_type']}: {explanation['formula']}")
```

---

## 🔍 Token Mappings

### Understanding Token Normalization

LLMCosts normalizes different provider token naming conventions:

```python
from llmcosts import get_token_mappings

# Get all mappings
mappings = get_token_mappings(include_examples=True)

# See how different providers map to normalized names
for mapping in mappings['token_mappings']:
    print(f"{mapping['normalized_name']}: {mapping['provider_aliases']}")
```

### Provider-Specific Mappings

```python
from llmcosts import get_provider_token_mappings, Provider

# Get OpenAI-specific mappings
openai_mappings = get_provider_token_mappings(Provider.OPENAI)

# See examples of normalization
for example in openai_mappings['examples']:
    print(f"Provider: {example['provider']}")
    print(f"Raw usage: {example['raw_usage']}")
    print(f"Normalized: {example['normalized_tokens']}")
    print(f"Explanation: {example['explanation']}")
```

---

## 🏗️ Error Handling

All functions may raise exceptions for:

- **401 Unauthorized**: Invalid API key
- **404 Not Found**: Resource not found
- **422 Unprocessable Entity**: Invalid request data
- **500 Internal Server Error**: Server error

```python
from llmcosts import LLMCostsClient, list_thresholds
from llmcosts.exceptions import TriggeredLimitError

try:
    client = LLMCostsClient(api_key="your-key")
    thresholds = list_thresholds(client)
except Exception as e:
    print(f"Error: {e}")
```

---

## 🎯 Best Practices

1. **Use Environment Variables**: Store API keys in environment variables
2. **Handle Errors Gracefully**: Always wrap API calls in try-catch blocks
3. **Filter Smartly**: Use filters to reduce API response sizes
4. **Cache Results**: Cache model lists and pricing data when possible
5. **Monitor Thresholds**: Set up alerts before hitting limits

---

## 📝 Complete Example

```python
from llmcosts import (
    LLMCostsClient,
    list_models,
    get_model_pricing,
    calculate_cost_from_tokens,
    list_thresholds,
    create_threshold,
    get_daily_costs,
    Provider
)

# Initialize client
client = LLMCostsClient()

# Discover models
models = list_models()
print(f"Available models: {len(models)}")

# Get pricing
pricing = get_model_pricing(Provider.OPENAI, "gpt-4o-mini")
print(f"Input cost: ${pricing['costs'][0]['cost_per_million']}/M tokens")

# Calculate cost
cost = calculate_cost_from_tokens(
    Provider.OPENAI,
    "gpt-4o-mini",
    input_tokens=1000,
    output_tokens=500
)
print(f"Estimated cost: ${cost['costs']['total_cost']}")

# Set up threshold
threshold = create_threshold(client, {
    "threshold_type": "alert",
    "amount": 100.0,
    "period": "month",
    "notification_list": ["email"]
})
print(f"Created threshold: {threshold['threshold_id']}")

# Check spending
daily_costs = get_daily_costs(start="2024-01-01", end="2024-01-31")
total_spent = sum(day['total_cost'] for day in daily_costs)
print(f"Total spent this month: ${total_spent}")
```

---

## 🔗 Related Documentation

- [Main README](../README.md) - Getting started and basic usage
- [Configuration Guide](configuration.md) - Advanced configuration options
- [Provider Guide](providers.md) - Provider-specific integration examples
- [Testing Guide](testing.md) - Testing and debugging
- [Troubleshooting](troubleshooting.md) - Common issues and solutions 