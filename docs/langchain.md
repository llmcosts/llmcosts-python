# LangChain Integration Guide

LLMCosts provides seamless integration with LangChain, automatically tracking usage for all your LangChain workflows including streaming, batch operations, and complex chains. **Zero code changes needed** - usage tracking happens automatically!

**🔒 Privacy Advantage**: Unlike LangChain's default behavior (which captures requests, responses, and more), LLMCosts NEVER sees your API keys, prompts, or responses. We only extract usage metadata (tokens, costs, model info) for cost tracking. Your data stays completely private.

**🔄 Universal Proxy**: The same `LLMTrackingProxy` works with any LLM provider in LangChain - OpenAI, Anthropic, Google, and more. No need for provider-specific wrappers.

## 📦 Prerequisites

```bash
# Install LangChain OpenAI integration
pip install langchain-openai

# For LangChain core components
pip install langchain-core

# Install LLMCosts with LangChain support
pip install "llmcosts[langchain]"
```

## 🔑 Critical Integration Pattern

**⚠️ Important**: LangChain integration requires the `framework=Framework.LANGCHAIN` parameter:

```python
from llmcosts.tracker import LLMTrackingProxy, Provider, Framework
from langchain_openai import OpenAI, ChatOpenAI
import openai

# Step 1: Create tracked OpenAI client with LangChain framework integration
openai_client = openai.OpenAI(api_key="your-key")
tracked_client = LLMTrackingProxy(
    openai_client,
    provider=Provider.OPENAI,        # REQUIRED: The actual LLM provider (OpenAI)
    framework=Framework.LANGCHAIN,   # REQUIRED: Enables LangChain-specific features
)

# ✨ Why framework=Framework.LANGCHAIN is needed:
# - Enables automatic stream_options injection for OpenAI streaming
# - Allows seamless streaming without manual stream_options configuration
# - Optimizes usage tracking for LangChain's specific API patterns

# Step 2: Pass correct sub-clients to LangChain models
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

## 🎯 Supported Operations

All LangChain operations are fully supported with automatic usage tracking:

| Operation | OpenAI LLM | ChatOpenAI | Tracking |
|-----------|------------|------------|----------|
| **Non-streaming** | ✅ `.invoke()` | ✅ `.invoke()` | ✅ Automatic |
| **Streaming** | ✅ `.stream()` | ✅ `.stream()` | ✅ Automatic |
| **Batch** | ✅ `.batch()` | ✅ `.batch()` | ✅ Automatic |
| **Chains** | ✅ LCEL Chains | ✅ LCEL Chains | ✅ Automatic |
| **Async** | ✅ `.ainvoke()`, `.astream()` | ✅ `.ainvoke()`, `.astream()` | ✅ Automatic |

## 📝 Basic Usage Examples

```python
from llmcosts.tracker import LLMTrackingProxy, Provider, Framework
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.messages import HumanMessage
import openai

# Setup tracked client
openai_client = openai.OpenAI(api_key="your-key")
tracked_client = LLMTrackingProxy(
    openai_client,
    provider=Provider.OPENAI,
    framework=Framework.LANGCHAIN,
)

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

## 🔗 Chains and Advanced Patterns

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

## 📦 Batch Operations

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

## 🌊 Streaming Details

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

## 🔧 Advanced Configuration

```python
# Context tracking for LangChain usage
tracked_client = LLMTrackingProxy(
    openai_client,
    provider=Provider.OPENAI,        # REQUIRED: The LLM provider
    framework=Framework.LANGCHAIN,   # REQUIRED: Enable LangChain integration
    context={
        "framework": "langchain",
        "user_id": "user_123",
        "session_id": "session_456"
    }
)

# Response callbacks
def track_langchain_response(response):
    print(f"LangChain response tracked: {response.id if hasattr(response, 'id') else 'N/A'}")

tracked_client.response_callback = track_langchain_response

# Debug mode for troubleshooting
tracked_client.debug = True
tracked_client.sync_mode = True  # Wait for tracking completion
```

## 🔍 Usage Validation

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

## ⚠️ Important Considerations

1. **Batch Behavior Differences**:
   - **OpenAI Completions**: Multiple prompts = 1 API call = 1 usage log
   - **ChatOpenAI**: Multiple messages = Multiple API calls = Multiple usage logs

2. **LangChain Integration**:
   - Pass `framework=Framework.LANGCHAIN` when creating `LLMTrackingProxy`
   - This enables automatic `stream_options` injection for seamless streaming
   - Without this, streaming calls will raise validation errors

3. **Streaming Requirements**:
   - OpenAI streaming requires `stream_options={'include_usage': True}`
   - LLMCosts **automatically injects** this when LangChain mode is enabled

4. **Model Support**:
   - Works with all OpenAI models supported by LangChain
   - Currently supports OpenAI provider only via LangChain integration

## 🔧 Troubleshooting

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

## 🚀 Migration Guide

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
from llmcosts.tracker import LLMTrackingProxy, Provider, Framework
from langchain_openai import ChatOpenAI
import openai

client = openai.OpenAI(api_key="your-key")
tracked_client = LLMTrackingProxy(
    client,
    provider=Provider.OPENAI,
    framework=Framework.LANGCHAIN,
)  # Add this line
chat_model = ChatOpenAI(
    client=tracked_client.chat.completions,  # Change this line
    model="gpt-4o-mini",
    max_tokens=100
)

# Everything else stays exactly the same!
response = chat_model.invoke([HumanMessage(content="Hello")])
```

**That's it!** Your existing LangChain code now has complete usage tracking with zero changes to your business logic.

## Next Steps

- **[Client Tracking & Context Data](client-tracking.md)** - Track costs per client with rich context data
- **[Configuration](configuration.md)** - Advanced configuration options
- **[Pricing](pricing.md)** - Model discovery and cost calculation
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions 