# LLMCosts

[![PyPI version](https://badge.fury.io/py/llmcosts.svg)](https://badge.fury.io/py/llmcosts)
[![Python Support](https://img.shields.io/pypi/pyversions/llmcosts.svg)](https://pypi.org/project/llmcosts/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LLMCosts** is a comprehensive LLM cost tracking and management platform that helps developers and agencies monitor, analyze, and optimize their AI spending across all major providers. To get started, **[sign up for a free account at llmcosts.com](https://llmcosts.com)** to access real-time analytics, budget alerts, client billing tools, and accounting integrations. Once registered, create an API key from your dashboard to use with this Python SDK.

**Free tier includes 2 million tokens per month** with no credit card required. Paid plans start at $19/month for advanced features like forecasting, Xero/QuickBooks integration, and unlimited users.

A universal Python wrapper that intercepts LLM API responses and extracts usage information for comprehensive cost tracking. Works as a drop-in replacement for your existing LLM clients with zero code changes to your API calls.

**🎯 Supports**: OpenAI (any OpenAI-compatible APIs -- DeepSeek, Grok, etc.), Anthropic, Google Gemini, AWS Bedrock, and LangChain.

## 🚀 Quick Start

### Installation

```bash
# Core library only (minimal dependencies)
pip install llmcosts

# With specific providers
pip install llmcosts[openai]      # OpenAI + compatible APIs (DeepSeek, Grok, etc.)
pip install llmcosts[anthropic]   # Anthropic Claude
pip install llmcosts[google]      # Google Gemini
pip install llmcosts[bedrock]     # AWS Bedrock
pip install llmcosts[langchain]   # LangChain integration

# All providers at once
pip install llmcosts[all]

# Using uv (recommended)
uv add llmcosts                   # Core only
uv add llmcosts[openai]           # With OpenAI
uv add llmcosts[langchain]        # With LangChain
uv add llmcosts[all]              # All providers
```

### Basic Usage

> **🔑 CRITICAL: API Key Required**
>
> Before using LLMCosts, you **MUST** have an LLMCosts API key. **[Sign up for a free account at llmcosts.com](https://llmcosts.com)** to get your API key.
>
> You can provide your API key in two ways:
> - **Environment variable**: Set `LLMCOSTS_API_KEY` in your environment or `.env` file
> - **Direct parameter**: Pass `api_key="your-llmcosts-api-key"` to `LLMTrackingProxy`
>
> **Without an API key, none of the LLMCosts tracking will work!**

```python
import os
from llmcosts import LLMTrackingProxy, Provider
import openai

# Create OpenAI client
client = openai.OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Wrap with LLMCosts tracking
tracked_client = LLMTrackingProxy(
    client, 
    provider=Provider.OPENAI,
    # This is the default and can be omitted
    api_key=os.environ.get("LLMCOSTS_API_KEY"),
    debug=True
)

# Use exactly as before - zero changes to your API calls
response = tracked_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Usage automatically logged as structured JSON
```

**Environment Setup (.env file):**

Create a `.env` file in your project root:

```bash
# Your LLMCosts API key (required)
LLMCOSTS_API_KEY=your-llmcosts-api-key-here

# Your LLM provider API keys (add only the ones you need)
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
GOOGLE_API_KEY=your-google-api-key-here
DEEPSEEK_API_KEY=your-deepseek-api-key-here
XAI_API_KEY=your-xai-api-key-here

# AWS credentials (for Bedrock)
AWS_ACCESS_KEY_ID=your-aws-access-key-here
AWS_SECRET_ACCESS_KEY=your-aws-secret-key-here
```

> **💡 Recommended Pattern**: Always create `LLMTrackingProxy` directly - it handles global tracker creation, API key management, and background processing automatically. Avoid calling `get_usage_tracker()` unless you need advanced debugging.

The same utilities can be imported directly from the ``llmcosts`` package for
convenience:

```python
from llmcosts import get_usage_tracker, list_alerts, list_limits
```

## 📋 Key Features

- **🔄 Universal Compatibility**: Works with all major LLM providers
- **📊 Automatic Usage Tracking**: Captures tokens, costs, model info, and timestamps
- **🎛️ Dynamic Configuration**: Change settings on-the-fly without restarting
- **💾 Smart Delivery**: Resilient background delivery with retry logic
- **📝 Custom Context**: Add user/session tracking data to every request
- **🔔 Response Callbacks**: Built-in SQLite/text file callbacks plus custom handlers
- **🔍 Debug Mode**: Synchronous operation for testing and debugging
- **📤 Structured Output**: Clean JSON format for easy parsing
- **♻️ Auto-Recovery**: Automatically restarts failed delivery threads
- **🚫 Non-Intrusive**: Original API responses remain completely unchanged

## 🎯 Supported Providers

| Provider | Import | Provider Enum | Installation |
|----------|---------|---------------|-------------|
| **OpenAI** | `import openai` | `Provider.OPENAI` | `pip install llmcosts[openai]` |
| **Anthropic** | `import anthropic` | `Provider.ANTHROPIC` | `pip install llmcosts[anthropic]` |
| **Google Gemini** | `import google.genai` | `Provider.GOOGLE` | `pip install llmcosts[google]` |
| **AWS Bedrock** | `import boto3` | `Provider.AMAZON_BEDROCK` | `pip install llmcosts[bedrock]` |
| **DeepSeek** | `import openai` | `Provider.DEEPSEEK` | `pip install llmcosts[openai]` |
| **Grok/xAI** | `import openai` | `Provider.XAI` | `pip install llmcosts[openai]` |
| **LangChain** | `import langchain_openai` | `Provider.OPENAI` | `pip install llmcosts[langchain]` |

## 💻 Usage Examples

### OpenAI

```python
import os
from llmcosts.tracker import LLMTrackingProxy, Provider
import openai

client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
tracked_client = LLMTrackingProxy(
    client, 
    provider=Provider.OPENAI,
    api_key=os.environ.get("LLMCOSTS_API_KEY"),
)

# Standard chat completion
response = tracked_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)

# Streaming (requires stream_options for OpenAI)
stream = tracked_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Count to 10"}],
    stream=True,
    stream_options={"include_usage": True}
)
for chunk in stream:
    print(chunk.choices[0].delta.content, end="")
```

### Anthropic

```python
import os
from llmcosts.tracker import LLMTrackingProxy, Provider
import anthropic

client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)
tracked_client = LLMTrackingProxy(
    client, 
    provider=Provider.ANTHROPIC,
    api_key=os.environ.get("LLMCOSTS_API_KEY"),
)

response = tracked_client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=1000,
    messages=[{"role": "user", "content": "Hello, Claude!"}]
)
```

### Google Gemini

```python
import os
from llmcosts.tracker import LLMTrackingProxy, Provider
import google.genai as genai

client = genai.Client(
    api_key=os.environ.get("GOOGLE_API_KEY"),
)
tracked_client = LLMTrackingProxy(
    client, 
    provider=Provider.GOOGLE,
    api_key=os.environ.get("LLMCOSTS_API_KEY"),
)

response = tracked_client.models.generate_content(
    model="gemini-pro",
    contents="Explain machine learning"
)
```

### AWS Bedrock

```python
import os
from llmcosts.tracker import LLMTrackingProxy, Provider
import boto3
import json

client = boto3.client(
    'bedrock-runtime',
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    region_name="us-east-1"
)
tracked_client = LLMTrackingProxy(
    client, 
    provider=Provider.AMAZON_BEDROCK,
    api_key=os.environ.get("LLMCOSTS_API_KEY"),
)

response = tracked_client.invoke_model(
    modelId="anthropic.claude-3-haiku-20240307-v1:0",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": "Hello!"}]
    })
)
```

### OpenAI-Compatible APIs

```python
import os
from llmcosts.tracker import LLMTrackingProxy, Provider
import openai

# DeepSeek
deepseek_client = openai.OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)
tracked_deepseek = LLMTrackingProxy(
    deepseek_client, 
    provider=Provider.DEEPSEEK,
    api_key=os.environ.get("LLMCOSTS_API_KEY"),
)

# Grok/xAI
grok_client = openai.OpenAI(
    api_key=os.environ.get("XAI_API_KEY"), 
    base_url="https://api.x.ai/v1"
)
tracked_grok = LLMTrackingProxy(
    grok_client, 
    provider=Provider.XAI,
    api_key=os.environ.get("LLMCOSTS_API_KEY"),
)
```

### LangChain Integration

LLMCosts provides seamless integration with LangChain, automatically tracking usage for all your LangChain workflows including streaming, batch operations, and complex chains. **Zero code changes needed** - usage tracking happens automatically!

#### 📦 Prerequisites

```bash
# Install LangChain OpenAI integration
pip install langchain-openai

# For LangChain core components
pip install langchain-core
```

#### 🔑 Critical Integration Pattern

**⚠️ Important**: LangChain models require **sub-clients**, not the full tracked client:

```python
from llmcosts.tracker import LLMTrackingProxy, Provider
from langchain_openai import OpenAI, ChatOpenAI
import openai

# Step 1: Create tracked OpenAI client  
openai_client = openai.OpenAI(api_key="your-key")
tracked_client = LLMTrackingProxy(openai_client, provider=Provider.OPENAI)

# Step 2: Enable LangChain compatibility mode
tracked_client.enable_langchain_mode()

# Step 3: Pass correct sub-clients to LangChain models
llm = OpenAI(
    client=tracked_client.completions,  # ✅ Use .completions for OpenAI LLM
    model="gpt-3.5-turbo-instruct",
    max_tokens=100,
    temperature=0.1,
)

chat_model = ChatOpenAI(
    client=tracked_client.chat.completions,  # ✅ Use .chat.completions for ChatOpenAI
    model="gpt-4o-mini",
    max_tokens=100,
    temperature=0.1,
)
```

> **💡 Why Sub-clients?** LangChain expects specific client interfaces. Using `tracked_client.completions` for OpenAI LLM and `tracked_client.chat.completions` for ChatOpenAI ensures compatibility while maintaining full usage tracking.

#### 🎯 Supported Operations

All LangChain operations are fully supported with automatic usage tracking:

| Operation | OpenAI LLM | ChatOpenAI | Tracking |
|-----------|------------|------------|----------|
| **Non-streaming** | ✅ `.invoke()` | ✅ `.invoke()` | ✅ Automatic |
| **Streaming** | ✅ `.stream()` | ✅ `.stream()` | ✅ Automatic |
| **Batch** | ✅ `.batch()` | ✅ `.batch()` | ✅ Automatic |
| **Chains** | ✅ LCEL Chains | ✅ LCEL Chains | ✅ Automatic |
| **Async** | ✅ `.ainvoke()`, `.astream()` | ✅ `.ainvoke()`, `.astream()` | ✅ Automatic |

#### 📝 Basic Usage Examples

```python
from llmcosts.tracker import LLMTrackingProxy, Provider
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.messages import HumanMessage
import openai

# Setup tracked client
openai_client = openai.OpenAI(api_key="your-key")
tracked_client = LLMTrackingProxy(openai_client, provider=Provider.OPENAI)
tracked_client.enable_langchain_mode()  # Enable LangChain compatibility

# OpenAI LLM (legacy completions)
llm = OpenAI(
    client=tracked_client.completions,
    model="gpt-3.5-turbo-instruct",
    max_tokens=100,
    temperature=0.1,
)

# Non-streaming
response = llm.invoke("Tell me a joke about programming")
print(response)

# Streaming
print("Streaming response:")
for chunk in llm.stream("Count from 1 to 5"):
    print(chunk, end="", flush=True)

# ChatOpenAI (chat completions)
chat_model = ChatOpenAI(
    client=tracked_client.chat.completions,
    model="gpt-4o-mini", 
    max_tokens=100,
    temperature=0.1,
)

# Non-streaming chat
messages = [HumanMessage(content="Explain quantum computing in one sentence")]
response = chat_model.invoke(messages)
print(f"Response: {response.content}")

# Streaming chat
print("Streaming chat:")
for chunk in chat_model.stream(messages):
    print(chunk.content, end="", flush=True)
```

#### 🔗 Chains and Advanced Patterns

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Create a chain
prompt = ChatPromptTemplate.from_messages([
    ("human", "Tell me a {length} fact about {topic}")
])

# Chain components together
chain = prompt | chat_model | StrOutputParser()

# Non-streaming chain
result = chain.invoke({"topic": "space", "length": "short"})
print(f"Chain result: {result}")

# Streaming chain  
print("Streaming chain:")
for chunk in chain.stream({"topic": "ocean", "length": "interesting"}):
    print(chunk, end="", flush=True)

# More complex chain with multiple steps
from langchain_core.runnables import RunnableLambda

def process_response(text):
    return f"Processed: {text.upper()}"

complex_chain = (
    prompt 
    | chat_model 
    | StrOutputParser() 
    | RunnableLambda(process_response)
)

result = complex_chain.invoke({"topic": "AI", "length": "brief"})
print(f"Complex chain: {result}")
```

#### 📦 Batch Operations

```python
# Batch completions (single API call for multiple prompts)
completion_prompts = ["What is Python?", "What is JavaScript?", "What is Go?"]
completion_responses = llm.batch(completion_prompts)
for i, response in enumerate(completion_responses):
    print(f"Q{i+1}: {completion_prompts[i]}")
    print(f"A{i+1}: {response}\n")

# Batch chat completions (separate API calls)
chat_messages = [
    [HumanMessage(content="What is machine learning?")],
    [HumanMessage(content="What is deep learning?")],
    [HumanMessage(content="What is neural network?")]
]

chat_responses = chat_model.batch(chat_messages)
for i, response in enumerate(chat_responses):
    print(f"Chat {i+1}: {response.content}")

# Batch chains
chain_inputs = [
    {"topic": "stars", "length": "short"},
    {"topic": "planets", "length": "brief"},
    {"topic": "galaxies", "length": "concise"}
]

chain_responses = chain.batch(chain_inputs)
for i, response in enumerate(chain_responses):
    print(f"Chain {i+1}: {response}")
```

#### 🌊 Streaming Details

LangChain streaming works seamlessly with automatic stream options injection:

```python
# Streaming automatically handles OpenAI's stream_options requirement
# No need to manually add stream_options={'include_usage': True}

# Streaming with callbacks
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

streaming_chat = ChatOpenAI(
    client=tracked_client.chat.completions,
    model="gpt-4o-mini",
    max_tokens=50,
    temperature=0.1,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

response = streaming_chat.invoke([HumanMessage(content="Tell me about the ocean")])
# Output streams to stdout AND tracks usage automatically
```

#### 🔧 Advanced Configuration

```python
# Context tracking for LangChain usage
tracked_client = LLMTrackingProxy(
    openai_client, 
    provider=Provider.OPENAI,
    context={
        "framework": "langchain",
        "user_id": "user_123",
        "session_id": "session_456"
    }
)
tracked_client.enable_langchain_mode()  # Enable LangChain compatibility

# Response callbacks
def track_langchain_response(response):
    print(f"LangChain response tracked: {response.id if hasattr(response, 'id') else 'N/A'}")

tracked_client.response_callback = track_langchain_response

# Debug mode for troubleshooting
tracked_client.debug = True
tracked_client.sync_mode = True  # Wait for tracking completion
```

#### 🔍 Usage Validation

Verify your LangChain integration is working:

```python
import logging

# Enable debug logging to see usage tracking
logging.basicConfig(level=logging.DEBUG)
tracked_client.debug = True

# Make a test call
response = chat_model.invoke([HumanMessage(content="Test")])

# Look for usage logs like:
# DEBUG:root:[LLM costs] OpenAI usage → {"usage": {...}, "model_id": "gpt-4o-mini", ...}
```

#### ⚠️ Important Considerations

1. **Batch Behavior Differences**:
   - **OpenAI Completions**: Multiple prompts = 1 API call = 1 usage log
   - **ChatOpenAI**: Multiple messages = Multiple API calls = Multiple usage logs

2. **LangChain Mode Required**:
   - **Must call** `tracked_client.enable_langchain_mode()` before using with LangChain
   - This enables automatic `stream_options` injection for seamless streaming
   - Without this, streaming calls will raise validation errors

3. **Streaming Requirements**:
   - OpenAI streaming requires `stream_options={'include_usage': True}`
   - LLMCosts **automatically injects** this when LangChain mode is enabled

4. **Model Support**:
   - Works with all OpenAI models supported by LangChain
   - Currently supports OpenAI provider only via LangChain integration

#### 🔧 Troubleshooting

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

**No usage logs appearing**:
```python
# Enable debug mode to see what's happening
tracked_client.debug = True
tracked_client.sync_mode = True

# Check for logs like: "[LLM costs] OpenAI usage →"
```

#### 🚀 Migration Guide

If you have existing LangChain code, here's how to add tracking:

```python
# Before (existing LangChain code)
from langchain_openai import ChatOpenAI
import openai

client = openai.OpenAI(api_key="your-key")
chat_model = ChatOpenAI(
    client=client,  # or openai_api_key="your-key"
    model="gpt-4o-mini",
    max_tokens=100
)

# After (with LLMCosts tracking)
from llmcosts.tracker import LLMTrackingProxy, Provider
from langchain_openai import ChatOpenAI
import openai

client = openai.OpenAI(api_key="your-key")
tracked_client = LLMTrackingProxy(client, provider=Provider.OPENAI)  # Add this line
tracked_client.enable_langchain_mode()  # Add this line
chat_model = ChatOpenAI(
    client=tracked_client.chat.completions,  # Change this line
    model="gpt-4o-mini",
    max_tokens=100
)

# Everything else stays exactly the same!
response = chat_model.invoke([HumanMessage(content="Hello")])
```

**That's it!** Your existing LangChain code now has complete usage tracking with zero changes to your business logic.

## 🔍 Discovering Supported Models and Providers

LLMCosts provides comprehensive SDK functions to discover which models and providers are supported. This is especially useful for validating configurations and building dynamic model selection interfaces.

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

We monitor model releases and update our database regularly, but for immediate needs, we're here to help!

## 💰 Pricing and Cost Calculation

LLMCosts provides comprehensive pricing information and cost calculation capabilities, allowing you to get real-time pricing data and calculate costs for your LLM usage.

### Model Pricing

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

### Token Mappings

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

### Cost Calculation

Calculate costs from token counts or raw usage data:

#### Calculate from Token Counts

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

#### Calculate from Raw Usage Data

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

### Real-World Usage Examples

#### Budget Estimation

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

#### Model Comparison

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

#### Integration with Tracking

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

### Using Models Functions with Tracker

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

## 🔧 Configuration

### LLMTrackingProxy Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target` | Any | Required | The LLM client to wrap |
| `provider` | Provider | Required | Provider enum specifying the LLM service |
| `debug` | bool | `False` | Enable debug logging |
| `sync_mode` | bool | `False` | Wait for usage tracker (good for testing) |
| `remote_save` | bool | `True` | Save usage events to remote server |
| `context` | dict | `None` | Custom context data for tracking |
| `response_callback` | callable | `None` | Function to process responses |
| `api_key` | str | `None` | LLMCOSTS API key (uses env var if not provided) |
| `client_customer_key` | str | `None` | Customer key for multi-tenant applications |

### Dynamic Property Updates

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

### Advanced Configuration

#### Context Tracking

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
    provider=Provider.OPENAI,
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

#### Customer Key Tracking

For multi-tenant applications, you can track usage per customer using the `client_customer_key` parameter. This is especially useful for billing, quota management, and usage analytics per customer.

```python
# Method 1: Set customer key at initialization
tracked_client = LLMTrackingProxy(
    client,
    provider=Provider.OPENAI,
    client_customer_key="customer_123"
)

# All API calls will include the customer key
response = tracked_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)
# → Sends to server: {"client_customer_key": "customer_123", "model_id": "gpt-4o-mini", ...}

# Method 2: Update customer key dynamically  
tracked_client.client_customer_key = "customer_456"
response = tracked_client.chat.completions.create(
    model="gpt-4o-mini", 
    messages=[{"role": "user", "content": "Another request"}]
)
# → Sends to server: {"client_customer_key": "customer_456", "model_id": "gpt-4o-mini", ...}

# Method 3: Combine with context data (context is for other tracking metadata)
tracked_client = LLMTrackingProxy(
    client,
    provider=Provider.OPENAI,
    client_customer_key="customer_789",
    context={
        "user_id": "user_123", 
        "session_id": "session_456",
        "feature": "chat_completion"
    }
)
```

**Multi-Customer Service Example:**

```python
import time
from llmcosts.tracker import LLMTrackingProxy, Provider
import openai

class MultiTenantLLMService:
    def __init__(self, api_key):
        self.base_client = openai.OpenAI(api_key=api_key)
        self.tracked_client = LLMTrackingProxy(
            self.base_client,
            provider=Provider.OPENAI
        )
    
    def chat_for_customer(self, customer_id, user_id, message):
        # Set customer key for billing/analytics (separate from context)
        self.tracked_client.client_customer_key = customer_id
        
        # Context is for other metadata
        self.tracked_client.context = {
            "user_id": user_id,
            "timestamp": time.time()
        }
        
        return self.tracked_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": message}]
        )

# Usage tracking with customer separation
service = MultiTenantLLMService("your-api-key")

# First customer call
response1 = service.chat_for_customer("acme_corp", "user123", "Hello!")
# → Sends: {"client_customer_key": "acme_corp", "context": {"user_id": "user123", ...}, ...}

# Second customer call  
response2 = service.chat_for_customer("globex_inc", "user456", "Hi there!")
# → Sends: {"client_customer_key": "globex_inc", "context": {"user_id": "user456", ...}, ...}
```

**Usage Data with Customer Key:**

When a customer key is provided, it appears as a top-level field in usage records:

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
  "client_customer_key": "customer_123",
  "context": {
    "user_id": "user123",
    "session_id": "session456"
  }
}
```

**Key Features:**
- **Separate from Context**: Customer key is a top-level field, not part of context metadata
- **Flexible Values**: Customer key can be any string, null, or empty
- **Automatic Inclusion**: Added to all usage records when set on the proxy
- **Dynamic Updates**: Change customer key mid-session with the setter property
- **Query Support**: Retrieve usage data filtered by customer key via the LLMCosts API
- **Billing Integration**: Perfect for customer-specific billing and quota tracking

#### Response Callbacks

LLMCosts includes built-in response callbacks for common use cases, and supports custom callbacks for specialized needs.

##### Built-in Callbacks

The package includes two ready-to-use callbacks for recording response data:

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

**Environment Setup for Built-in Callbacks:**

Create a `.env` file or set environment variables:

```bash
# For SQLite callback
export SQLITE_CALLBACK_TARGET_PATH="./data"

# For text file callback  
export TEXT_CALLBACK_TARGET_PATH="./logs"
```

**What Gets Recorded:**

Both callbacks extract and store cost event data from LLM responses:

```json
{
  "response_id": "chatcmpl-123abc",
  "model_id": "gpt-4o-mini",
  "provider": "openai",
  "timestamp": "2024-01-15T10:30:00Z",
  "input_tokens": 50,
  "output_tokens": 150,
  "total_tokens": 200,
  "context": {"user_id": "123", "session": "abc"},
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

- **SQLite**: Records are stored in a `cost_events` table with automatic schema creation
- **Text File**: Records are stored in JSON Lines format (one JSON object per line)
- **Record Overwriting**: Both callbacks overwrite records with the same `response_id`
- **Error Handling**: Robust error handling ensures callbacks don't interrupt main API calls

##### Custom Callbacks

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

**Advanced: Multiple Callbacks**

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

#### Testing and Debugging

```python
# Enable synchronous mode for testing
tracked_client = LLMTrackingProxy(
    client,
    provider=Provider.OPENAI,
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

Usage data is logged as structured JSON:

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

- **usage**: Token/unit counts (varies by provider)
- **model_id**: Model identifier used for the request
- **response_id**: Unique response identifier
- **timestamp**: ISO 8601 timestamp
- **provider**: LLM provider name
- **service_tier**: Service tier (when available)
- **context**: Custom tracking data
- **remote_save**: Only included when `false`

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

**Note:** The `LLMCOSTS_API_ENDPOINT` environment variable is only read when creating a new tracker instance. To use a custom endpoint, you must set the environment variable before creating any `LLMTrackingProxy` instances or call `reset_global_tracker()` to force recreation of the global tracker.

```python
from llmcosts.tracker import reset_global_tracker
import os

# Change endpoint and force tracker recreation
os.environ["LLMCOSTS_API_ENDPOINT"] = "https://your-endpoint.com/api/v1/usage"
reset_global_tracker()  # Force new tracker with custom endpoint

# New proxy instances will now use the custom endpoint
tracked_client = LLMTrackingProxy(client, provider=Provider.OPENAI)
```

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

## 🧪 Testing

LLMCosts includes comprehensive testing for all supported LLM providers, endpoint integration, and response callbacks. The test suite supports both automated testing and manual validation modes.

### Quick Start

1. **Install test dependencies:**
   ```bash
   # Install test dependencies using uv (recommended)
   uv sync --extra test
   
   # Or install test dependencies only
   uv pip install -e ".[test]"
   
   # Or using pip with requirements file
   pip install -r requirements-test.txt
   ```

2. **Copy environment file:**
   ```bash
   cp tests/env.example tests/.env
   ```

3. **Add your API keys to `tests/.env`**

4. **Run tests:**
   ```bash
   # Quick manual test
   uv run python tests/check.py openai gpt-4o-mini
   
   # Full test suite
   uv run python tests/check.py --test
   
   # Run specific test files (requires test dependencies)
   uv run pytest tests/test_openai_nonstreaming.py -v
   
   # Run all tests
   uv run pytest
   ```

**⚠️ Important**: Always use `uv run pytest` instead of `pytest` directly to ensure proper dependency management.

### Test Dependencies

The test suite requires additional dependencies for full provider support:

- **`boto3`** - For AWS Bedrock tests
- **`langchain`** and **`langchain-openai`** - For LangChain integration tests
- **`pytest`** and **`pytest-cov`** - For test execution and coverage

These are automatically installed when using `uv sync --extra test` or `uv pip install -e ".[test]"`.

For comprehensive testing documentation, including callback testing, provider-specific tests, and advanced debugging, see **[tests/README_ENDPOINT_TESTING.md](tests/README_ENDPOINT_TESTING.md)**.

## 🛠️ Development

### Setup

```bash
# Clone repository
git clone https://github.com/keytonweissinger/llmcosts.git
cd llmcosts

# Using uv (recommended)
uv sync --extra dev

# Using pip
pip install -e ".[dev]"
```

### Code Quality

```bash
# Format code
uv run black llmcosts/ tests/
uv run isort llmcosts/ tests/

# Type checking
uv run mypy llmcosts/

# Run tests (requires test dependencies)
uv run pytest

# Run tests with coverage
uv run pytest --cov=llmcosts --cov-report=html
```

## 🐛 Troubleshooting

### Common Issues

**1. Missing API Key Error**
```
ValueError: LLMCOSTS_API_KEY is required
```
**Solution**: Set the `LLMCOSTS_API_KEY` environment variable or pass `api_key` parameter.

**2. OpenAI Streaming Without Usage**
```
stream_options={"include_usage": True} required for OpenAI streaming
```
**Solution**: Add `stream_options={"include_usage": True}` to OpenAI streaming calls.

**3. Tracker Not Starting**
```python
# Check tracker health

from llmcosts import get_usage_tracker
tracker = get_usage_tracker()
health = tracker.get_health_info()
print(health)
```

**4. Queue Full Warnings**
```
Usage queue is full. Dropping usage data.
```
**Solution**: Increase `max_queue_size` or check network connectivity.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run the test suite: `uv run pytest`
5. Ensure code quality: `uv run black llmcosts/ tests/` and `uv run isort llmcosts/ tests/`
6. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **PyPI**: [https://pypi.org/project/llmcosts/](https://pypi.org/project/llmcosts/)
- **GitHub**: [https://github.com/keytonweissinger/llmcosts](https://github.com/keytonweissinger/llmcosts)
- **Issues**: [https://github.com/keytonweissinger/llmcosts/issues](https://github.com/keytonweissinger/llmcosts/issues)
- **Documentation**: This README

## 📈 Changelog

### v0.1.0 (Current)
- Universal LLM provider support
- Dynamic configuration with property setters
- Context tracking for user/session data
- Response callbacks for custom processing
- Synchronous mode for testing
- Resilient background delivery
- Comprehensive test coverage
- Thread-safe global tracker management
