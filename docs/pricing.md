# Pricing and Cost Calculation Guide

LLMCosts provides comprehensive pricing information and cost calculation capabilities, allowing you to get real-time pricing data and calculate costs for your LLM usage.

**🔄 Universal Compatibility**: The same pricing and model discovery functions work with ANY LLM provider. One SDK, all providers.

**🔒 Privacy-First**: Model discovery and cost calculations happen locally using our pricing database. No sensitive data is transmitted during these operations.

## 🔍 Model Discovery

### Models SDK Functions

```python
from llmcosts import (
    list_models,
    get_models_dict,
    get_models_by_provider,
    get_providers_by_model,
    is_model_supported,
    Provider
)

# Get all available models
all_models = list_models()
print(f"Total models available: {len(all_models)}")
for model in all_models[:3]:  # Show first 3
    print(f"  {model['provider']}: {model['model_id']} (aliases: {model['aliases']})")

# Get models organized by provider
models_by_provider = get_models_dict()
print(f"Available providers: {list(models_by_provider.keys())}")

# Get all models for a specific provider
openai_models = get_models_by_provider(Provider.OPENAI)
print(f"OpenAI models: {openai_models[:5]}")  # Show first 5

# Using string provider names (case-insensitive)
anthropic_models = get_models_by_provider("anthropic")
google_models = get_models_by_provider("GOOGLE")  # Case doesn't matter

# Find which providers support a specific model
gpt4_providers = get_providers_by_model("gpt-4")
print(f"GPT-4 supported by: {gpt4_providers}")

# Check if a provider/model combination is supported
if is_model_supported(Provider.OPENAI, "gpt-4o-mini"):
    print("✅ OpenAI supports GPT-4o Mini")

# Works with model aliases too
if is_model_supported("anthropic", "claude-3-sonnet"):
    print("✅ Anthropic supports Claude 3 Sonnet")

# Validate before creating tracker
model = "gpt-4"
provider = "openai"
if is_model_supported(provider, model):
    # Safe to create tracker
    tracked_client = LLMTrackingProxy(client, provider=Provider.OPENAI)
else:
    print(f"❌ {provider} doesn't support {model}")
```

### 🚨 Need a Model or Provider Added?

**Contact [help@llmcosts.com](mailto:help@llmcosts.com) and we'll add it within 24 hours!**

We actively maintain our model database and add new providers and models quickly. Don't wait - if you need support for a new model or provider, just let us know and we'll get it set up fast. Include:

- Provider name (e.g., "Cohere", "Mistral", "OpenRouter")
- Model IDs you need (e.g., "command-r-plus", "mistral-large")
- Any aliases or alternative names
- API documentation if it's a new provider

## 💰 Model Pricing

Get detailed pricing information for any supported model:

```python
from llmcosts import get_model_pricing, Provider

# Get pricing for a specific model
pricing = get_model_pricing(Provider.OPENAI, "gpt-4o-mini")
if pricing:
    print(f"Model: {pricing['model_id']}")
    print(f"Provider: {pricing['provider']}")
    for cost in pricing['costs']:
        print(f"  {cost['token_type']}: ${cost['cost_per_million']}/M tokens")

# Example output:
# Model: gpt-4o-mini
# Provider: openai
#   input: $0.15/M tokens
#   output: $0.6/M tokens

# Works with string provider names too
anthropic_pricing = get_model_pricing("anthropic", "claude-3-haiku-20240307")

# Works with model aliases
alias_pricing = get_model_pricing(Provider.OPENAI, "gpt-4-turbo")
```

## 🔢 Token Mappings

Understand how different providers represent token usage:

```python
from llmcosts import get_token_mappings, get_provider_token_mappings, Provider

# Get token mappings for all providers
all_mappings = get_token_mappings()
print(f"Supported providers: {all_mappings['supported_providers']}")

# Get normalized token types
for mapping in all_mappings['token_mappings']:
    print(f"{mapping['normalized_name']}: {mapping['description']}")
    print(f"  Provider aliases: {mapping['provider_aliases']}")

# Get mappings for a specific provider with examples
openai_mappings = get_token_mappings(Provider.OPENAI, include_examples=True)
for example in openai_mappings['examples']:
    print(f"Raw OpenAI usage: {example['raw_usage']}")
    print(f"Normalized tokens: {example['normalized_tokens']}")
    print(f"Explanation: {example['explanation']}")

# Get detailed mappings for a specific provider
provider_mappings = get_provider_token_mappings(Provider.ANTHROPIC)
print(f"Anthropic token mappings: {len(provider_mappings['token_mappings'])} types")
```

## 📊 Cost Calculation

Calculate costs from token counts or raw usage data:

### Calculate from Token Counts

```python
from llmcosts import calculate_cost_from_tokens, Provider

# Calculate cost using normalized token counts
cost_result = calculate_cost_from_tokens(
    provider=Provider.OPENAI,
    model_id="gpt-4o-mini",
    input_tokens=1000,
    output_tokens=500,
    include_explanation=True
)

print(f"Total cost: ${cost_result['costs']['total_cost']}")
print(f"Input cost: ${cost_result['costs']['input_cost']}")
print(f"Output cost: ${cost_result['costs']['output_cost']}")

# With detailed explanations
if cost_result['explanations']:
    for explanation in cost_result['explanations']:
        print(f"{explanation['token_type']}: {explanation['formula']}")
        print(f"  Rate: ${explanation['rate_per_million']}/M tokens")
        print(f"  Count: {explanation['raw_count']} tokens")
        print(f"  Cost: ${explanation['calculated_cost']}")

# Calculate with all token types (cache, reasoning, etc.)
advanced_cost = calculate_cost_from_tokens(
    provider=Provider.OPENAI,
    model_id="gpt-4o-mini",
    input_tokens=1000,
    output_tokens=500,
    cache_read_tokens=100,
    cache_write_tokens=50,
    reasoning_tokens=200,  # For o1 models
    tool_use_tokens=25
)
```

### Calculate from Raw Usage Data

```python
from llmcosts import calculate_cost_from_usage, Provider

# Calculate cost from OpenAI response usage
openai_usage = {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150
}

cost_result = calculate_cost_from_usage(
    provider=Provider.OPENAI,
    model_id="gpt-4o-mini",
    usage=openai_usage
)

print(f"Cost from OpenAI usage: ${cost_result['costs']['total_cost']}")

# Calculate cost from Anthropic response usage
anthropic_usage = {
    "input_tokens": 100,
    "output_tokens": 50
}

cost_result = calculate_cost_from_usage(
    provider=Provider.ANTHROPIC,
    model_id="claude-3-haiku-20240307",
    usage=anthropic_usage,
    include_explanation=True
)

print(f"Cost from Anthropic usage: ${cost_result['costs']['total_cost']}")
```

## 💡 Real-World Usage Examples

### Budget Estimation

```python
from llmcosts import get_model_pricing, calculate_cost_from_tokens, Provider

def estimate_monthly_budget(provider, model_id, daily_tokens):
    """Estimate monthly costs based on daily token usage."""
    pricing = get_model_pricing(provider, model_id)
    if not pricing:
        return None
    
    # Calculate daily cost
    daily_cost = calculate_cost_from_tokens(
        provider=provider,
        model_id=model_id,
        input_tokens=daily_tokens['input'],
        output_tokens=daily_tokens['output']
    )
    
    monthly_cost = daily_cost['costs']['total_cost'] * 30
    return {
        'model': model_id,
        'daily_cost': daily_cost['costs']['total_cost'],
        'monthly_cost': monthly_cost,
        'pricing_breakdown': pricing['costs']
    }

# Usage
budget = estimate_monthly_budget(
    Provider.OPENAI, 
    "gpt-4o-mini",
    {"input": 50000, "output": 25000}  # 50K input, 25K output per day
)

if budget:
    print(f"Monthly budget for {budget['model']}: ${budget['monthly_cost']:.2f}")
```

### Model Comparison

```python
from llmcosts import get_model_pricing, calculate_cost_from_tokens, Provider

def compare_model_costs(models, input_tokens, output_tokens):
    """Compare costs across different models."""
    comparisons = []
    
    for provider, model_id in models:
        cost_result = calculate_cost_from_tokens(
            provider=provider,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
        
        if cost_result['model_found']:
            comparisons.append({
                'provider': provider,
                'model': model_id,
                'total_cost': cost_result['costs']['total_cost'],
                'input_cost': cost_result['costs']['input_cost'],
                'output_cost': cost_result['costs']['output_cost']
            })
    
    return sorted(comparisons, key=lambda x: x['total_cost'])

# Compare costs for the same task
models_to_compare = [
    (Provider.OPENAI, "gpt-4o-mini"),
    (Provider.ANTHROPIC, "claude-3-haiku-20240307"),
    (Provider.OPENAI, "gpt-4o"),
]

comparison = compare_model_costs(models_to_compare, 1000, 500)
print("\nModel cost comparison (1000 input, 500 output tokens):")
for i, model in enumerate(comparison):
    print(f"{i+1}. {model['provider']} {model['model']}: ${model['total_cost']:.4f}")
```

### Integration with Tracking

```python
from llmcosts import LLMTrackingProxy, get_model_pricing, calculate_cost_from_tokens, Provider
import openai

class CostAwareLLMClient:
    """LLM client with cost estimation and tracking."""
    
    def __init__(self, api_key, llmcosts_api_key):
        self.client = openai.OpenAI(api_key=api_key)
        self.tracked_client = LLMTrackingProxy(
            self.client, 
            provider=Provider.OPENAI,
            api_key=llmcosts_api_key
        )
    
    def estimate_cost_before_call(self, model, messages):
        """Estimate cost before making the API call."""
        # Rough estimation based on message content
        estimated_input_tokens = sum(len(msg['content']) // 4 for msg in messages)
        estimated_output_tokens = estimated_input_tokens // 2  # Rough estimate
        
        cost_estimate = calculate_cost_from_tokens(
            provider=Provider.OPENAI,
            model_id=model,
            input_tokens=estimated_input_tokens,
            output_tokens=estimated_output_tokens
        )
        
        return cost_estimate['costs']['total_cost']
    
    def chat_with_cost_info(self, model, messages, max_cost=None):
        """Make a chat call with cost estimation and tracking."""
        # Pre-call cost estimation
        estimated_cost = self.estimate_cost_before_call(model, messages)
        
        if max_cost and estimated_cost > max_cost:
            raise ValueError(f"Estimated cost ${estimated_cost:.4f} exceeds limit ${max_cost:.4f}")
        
        print(f"Estimated cost: ${estimated_cost:.4f}")
        
        # Make the tracked call
        response = self.tracked_client.chat.completions.create(
            model=model,
            messages=messages
        )
        
        # Calculate actual cost
        actual_cost = calculate_cost_from_usage(
            provider=Provider.OPENAI,
            model_id=model,
            usage=response.usage.__dict__
        )
        
        print(f"Actual cost: ${actual_cost['costs']['total_cost']:.4f}")
        
        return response, actual_cost

# Usage
cost_aware = CostAwareLLMClient("your-openai-key", "your-llmcosts-key")

response, cost_info = cost_aware.chat_with_cost_info(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    max_cost=0.01  # 1 cent limit
)

print(f"Response: {response.choices[0].message.content[:100]}...")
print(f"Final cost breakdown: {cost_info['costs']}")
```

## 🔧 Using Models Functions with Tracker

Combine the models discovery functions with the tracker for robust applications:

```python
from llmcosts import LLMTrackingProxy, Provider, get_models_by_provider, is_model_supported
import openai

# Get available models before setup
available_models = get_models_by_provider(Provider.OPENAI)
print(f"Available OpenAI models: {available_models}")

# Validate model before using
model_to_use = "gpt-4o-mini"
if is_model_supported(Provider.OPENAI, model_to_use):
    client = openai.OpenAI(api_key="your-key")
    tracked_client = LLMTrackingProxy(client, provider=Provider.OPENAI)
    
    response = tracked_client.chat.completions.create(
        model=model_to_use,
        messages=[{"role": "user", "content": "Hello!"}]
    )
else:
    print(f"Model {model_to_use} not supported by OpenAI")

# Build dynamic model selector
def select_model_for_task(task_type: str, provider: Provider):
    """Example of dynamic model selection based on task."""
    available = get_models_by_provider(provider)
    
    if task_type == "reasoning" and provider == Provider.OPENAI:
        reasoning_models = [m for m in available if "o1" in m.lower()]
        return reasoning_models[0] if reasoning_models else available[0]
    elif task_type == "fast" and provider == Provider.OPENAI:
        fast_models = [m for m in available if "gpt-4o-mini" in m.lower()]
        return fast_models[0] if fast_models else available[0]
    else:
        return available[0] if available else None

# Use dynamic selection
model = select_model_for_task("reasoning", Provider.OPENAI)
if model:
    print(f"Selected model for reasoning: {model}")
```

## Next Steps

- **[Client Tracking & Context Data](client-tracking.md)** - Track costs per client with rich context data
- **[Configuration](configuration.md)** - Advanced configuration options
- **[Providers](providers.md)** - Provider-specific integration guides
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions 