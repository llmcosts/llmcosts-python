# Provider Integration Guide

This guide covers how to integrate LLMCosts with all supported LLM providers. **The same `LLMTrackingProxy` works with ANY LLM provider's SDK** - no need for different wrappers per provider.

**🔒 Privacy-First**: LLMCosts NEVER sees your API keys, requests, or responses. We only extract usage metadata (tokens, costs, model info) from the response objects. Your data stays private and secure.

## 🎯 Supported Providers

| Provider | Import | Provider Enum | Framework | Installation |
|----------|---------|---------------|-----------|-------------|
| **OpenAI** | `import openai` | `Provider.OPENAI` | `None` (default) | `pip install "llmcosts[openai]"` |
| **Anthropic** | `import anthropic` | `Provider.ANTHROPIC` | `None` (default) | `pip install "llmcosts[anthropic]"` |
| **Google Gemini** | `import google.genai` | `Provider.GOOGLE` | `None` (default) | `pip install "llmcosts[google]"` |
| **AWS Bedrock** | `import boto3` | `Provider.AMAZON_BEDROCK` | `None` (default) | `pip install "llmcosts[bedrock]"` |
| **DeepSeek** | `import openai` | `Provider.DEEPSEEK` | `None` (default) | `pip install "llmcosts[openai]"` |
| **Grok/xAI** | `import openai` | `Provider.XAI` | `None` (default) | `pip install "llmcosts[openai]"` |
| **LangChain + OpenAI** | `import langchain_openai` | `Provider.OPENAI` | `Framework.LANGCHAIN` | `pip install "llmcosts[langchain]"` |

**Note:** LangChain is a framework, not a provider. When using LangChain, you specify both the underlying provider (e.g., `Provider.OPENAI`) and the framework (`Framework.LANGCHAIN`).

## Provider vs Framework: Key Concepts

**🔑 Understanding the Distinction:**

- **Provider** (`provider=Provider.OPENAI`): The actual LLM service you're using
  - **Required for every tracking setup**
  - Examples: OpenAI, Anthropic, Google, AWS Bedrock, DeepSeek, Grok
  - Determines how usage data is extracted and costs are calculated

- **Framework** (`framework=Framework.LANGCHAIN`): Optional integration layer
  - **Usually `None` (default) for direct API usage**
  - Only needed for special framework integrations (currently: LangChain)
  - Enables framework-specific features like automatic stream options injection

**📋 When to Use Framework Parameter:**

| Usage Scenario | Provider | Framework | Example |
|----------------|----------|-----------|---------|
| **Direct OpenAI API** | `Provider.OPENAI` | `None` (default) | Regular `openai.OpenAI()` usage |
| **Direct Anthropic API** | `Provider.ANTHROPIC` | `None` (default) | Regular `anthropic.Anthropic()` usage |
| **LangChain + OpenAI** | `Provider.OPENAI` | `Framework.LANGCHAIN` | Using `langchain_openai.ChatOpenAI` |
| **LangChain + Anthropic** | `Provider.ANTHROPIC` | `Framework.LANGCHAIN` | Using `langchain_anthropic.ChatAnthropic` |
| **DeepSeek API** | `Provider.DEEPSEEK` | `None` (default) | OpenAI-compatible DeepSeek API |

## OpenAI

```python
import os
from llmcosts.tracker import LLMTrackingProxy, Provider
import openai

client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
tracked_client = LLMTrackingProxy(
    client, 
    provider=Provider.OPENAI,  # REQUIRED: Specifies this is OpenAI
    # framework=None by default for direct OpenAI API usage
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

## Anthropic

```python
import os
from llmcosts.tracker import LLMTrackingProxy, Provider
import anthropic

client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)
tracked_client = LLMTrackingProxy(
    client, 
    provider=Provider.ANTHROPIC,  # REQUIRED: Specifies this is Anthropic
    # framework=None by default for direct Anthropic API usage
    api_key=os.environ.get("LLMCOSTS_API_KEY"),
)

response = tracked_client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=1000,
    messages=[{"role": "user", "content": "Hello, Claude!"}]
)
```

## Google Gemini

```python
import os
from llmcosts.tracker import LLMTrackingProxy, Provider
import google.genai as genai

client = genai.Client(
    api_key=os.environ.get("GOOGLE_API_KEY"),
)
tracked_client = LLMTrackingProxy(
    client, 
    provider=Provider.GOOGLE,  # REQUIRED: Specifies this is Google
    # framework=None by default for direct Google API usage
    api_key=os.environ.get("LLMCOSTS_API_KEY"),
)

response = tracked_client.models.generate_content(
    model="gemini-pro",
    contents="Explain machine learning"
)
```

## AWS Bedrock

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
    provider=Provider.AMAZON_BEDROCK,  # REQUIRED: Specifies this is AWS Bedrock
    # framework=None by default for direct Bedrock API usage
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

## OpenAI-Compatible APIs

### DeepSeek

```python
import os
from llmcosts.tracker import LLMTrackingProxy, Provider
import openai

deepseek_client = openai.OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)
tracked_deepseek = LLMTrackingProxy(
    deepseek_client, 
    provider=Provider.DEEPSEEK,  # REQUIRED: Specifies this is DeepSeek
    # framework=None by default for direct DeepSeek API usage
    api_key=os.environ.get("LLMCOSTS_API_KEY"),
)

response = tracked_deepseek.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Hello DeepSeek!"}]
)
```

### Grok/xAI

```python
import os
from llmcosts.tracker import LLMTrackingProxy, Provider
import openai

grok_client = openai.OpenAI(
    api_key=os.environ.get("XAI_API_KEY"), 
    base_url="https://api.x.ai/v1"
)
tracked_grok = LLMTrackingProxy(
    grok_client, 
    provider=Provider.XAI,  # REQUIRED: Specifies this is Grok/xAI
    # framework=None by default for direct Grok API usage
    api_key=os.environ.get("LLMCOSTS_API_KEY"),
)

response = tracked_grok.chat.completions.create(
    model="grok-beta",
    messages=[{"role": "user", "content": "Hello Grok!"}]
)
```

## Environment Setup

LLMCosts only needs its own API key:

```bash
# Your LLMCosts API key (required) - get this from llmcosts.com
LLMCOSTS_API_KEY=your-llmcosts-api-key-here
```

**🔒 Your Provider API Keys Stay Private**: Continue using your existing LLM provider API keys exactly as before. LLMCosts never sees them - we only extract usage metadata from responses.

**Example provider key setup** (these stay yours):
```bash
# Examples - these remain private and are never shared with LLMCosts
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
GOOGLE_API_KEY=your-google-api-key-here
DEEPSEEK_API_KEY=your-deepseek-api-key-here
XAI_API_KEY=your-xai-api-key-here

# AWS credentials (for Bedrock)
AWS_ACCESS_KEY_ID=your-aws-access-key-here
AWS_SECRET_ACCESS_KEY=your-aws-secret-key-here
```

## Next Steps

- **[Client Tracking & Context Data](client-tracking.md)** - Track costs per client with rich context data
- **[LangChain Integration](langchain.md)** - Detailed LangChain integration guide
- **[Configuration](configuration.md)** - Advanced configuration options
- **[Pricing](pricing.md)** - Model discovery and cost calculation
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions 