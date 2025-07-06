# Testing Guide

LLMCosts includes comprehensive testing for all supported LLM providers, endpoint integration, and response callbacks. The test suite supports both automated testing and manual validation modes.

**🔒 Privacy During Testing**: All tests maintain our privacy-first principles. Even during testing, only usage metadata is processed - never API keys, requests, or responses.

**🔄 Universal Test Framework**: The same test patterns work across ANY LLM provider since we use one universal tracking proxy. Add new providers without changing test architecture.

## 🚀 Quick Start

### 1. Install Test Dependencies

```bash
# Install test dependencies using uv (recommended)
uv sync --extra test

# Or install test dependencies only
uv pip install -e ".[test]"

# Or using pip with requirements file
pip install -e ".[test]"
```

### 2. Environment Setup

```bash
# Copy environment template
cp tests/env.example tests/.env

# Edit tests/.env with your API keys
```

### 3. Run Tests

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

## 🔧 Test Dependencies

The test suite requires additional dependencies for full provider support:

- **`boto3`** - For AWS Bedrock tests
- **`langchain`** and **`langchain-openai`** - For LangChain integration tests
- **`pytest`** and **`pytest-cov`** - For test execution and coverage

These are automatically installed when using `uv sync --extra test` or `uv pip install -e ".[test]"`.

## 📋 Test Categories

### Provider Tests

Each provider has dedicated test files:

- **OpenAI**: `test_openai_nonstreaming.py`, `test_openai_streaming.py`
- **Anthropic**: `test_anthropic_nonstreaming.py`, `test_anthropic_streaming.py`
- **Google**: `test_gemini_nonstreaming.py`, `test_gemini_streaming.py`
- **AWS Bedrock**: `test_bedrock_nonstreaming.py`, `test_bedrock_streaming.py`
- **DeepSeek**: `test_deepseek_nonstreaming.py`, `test_deepseek_streaming.py`
- **Grok/xAI**: `test_grok_nonstreaming.py`, `test_grok_streaming.py`

### Framework Tests

- **LangChain**: `test_langchain_nonstreaming.py`, `test_langchain_streaming.py`

### Feature Tests

- **Pricing**: `test_pricing_and_costs.py`
- **Models**: `test_models.py`
- **Callbacks**: `test_sqlite_callback.py`, `test_text_callback.py`
- **Limits**: `test_*_limit.py` files
- **Threading**: `test_thread_safety.py`

## 🎯 Manual Testing

### Quick Provider Check

```bash
# Test OpenAI
uv run python tests/check.py openai gpt-4o-mini

# Test Anthropic
uv run python tests/check.py anthropic claude-3-haiku-20240307

# Test Google
uv run python tests/check.py google gemini-1.5-flash

# Test AWS Bedrock
uv run python tests/check.py bedrock anthropic.claude-3-haiku-20240307-v1:0

# Test DeepSeek
uv run python tests/check.py deepseek deepseek-chat

# Test Grok
uv run python tests/check.py xai grok-beta
```

### Manual Streaming Tests

```bash
# Located in tests/manual/
uv run python tests/manual/manual_check_streaming.py
uv run python tests/manual/manual_check_nonstreaming.py
```

## 🔬 Automated Testing

### Run All Tests

```bash
# Run full test suite
uv run pytest

# Run with coverage
uv run pytest --cov=llmcosts --cov-report=html

# Run specific test categories
uv run pytest tests/test_openai_* -v
uv run pytest tests/test_*_streaming.py -v
uv run pytest tests/test_*_limit.py -v
```

### Test Configuration

**🔒 Privacy Note**: These API keys are for YOUR LOCAL TESTING ONLY. LLMCosts never sees or requires your provider API keys - they stay on your machine for testing purposes.

Tests use configuration from `tests/.env`:

```bash
# Your LLMCosts API key (required for all tests)
LLMCOSTS_API_KEY=your-llmcosts-api-key

# YOUR provider API keys (for local testing only - never shared with LLMCosts)
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key
DEEPSEEK_API_KEY=your-deepseek-key
XAI_API_KEY=your-xai-key

# AWS credentials (for Bedrock tests - stays local)
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_DEFAULT_REGION=us-east-1
```

**Why do tests need provider API keys?**
- Tests make real API calls to verify LLMCosts works correctly with each provider
- These keys remain on YOUR machine - they're never transmitted to LLMCosts
- LLMCosts only extracts usage metadata from the responses
- Skip tests for providers you don't have keys for

## 📊 Callback Testing

### SQLite Callback Tests

```bash
# Run SQLite callback tests
uv run pytest tests/test_sqlite_callback.py -v

# Manual SQLite callback test
uv run python tests/test_sqlite_callback.py
```

### Text Callback Tests

```bash
# Run text callback tests
uv run pytest tests/test_text_callback.py -v

# Manual text callback test
uv run python tests/test_text_callback.py
```

### Setup for Callback Tests

```bash
# Set environment variables for callback tests
export SQLITE_CALLBACK_TARGET_PATH="./test_data"
export TEXT_CALLBACK_TARGET_PATH="./test_logs"

# Or add to tests/.env
echo "SQLITE_CALLBACK_TARGET_PATH=./test_data" >> tests/.env
echo "TEXT_CALLBACK_TARGET_PATH=./test_logs" >> tests/.env
```

## 🏗️ Integration Testing

### Endpoint Integration

```bash
# Test API endpoint integration
uv run pytest tests/test_endpoint_integration.py -v

# Manual endpoint test
uv run python tests/test_endpoint_integration.py
```

### Framework Integration

```bash
# Test LangChain integration
uv run pytest tests/test_langchain_* -v

# Test framework parameter handling
uv run pytest tests/test_framework_parameter.py -v
```

## 🔧 Writing Custom Tests

### Basic Test Structure

```python
import os
import pytest
from llmcosts.tracker import LLMTrackingProxy, Provider
import openai

class TestCustomProvider:
    def setup_method(self):
        """Setup for each test method."""
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            pytest.skip("OPENAI_API_KEY not set")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.tracked_client = LLMTrackingProxy(
            self.client,
            provider=Provider.OPENAI,
            debug=True,
            sync_mode=True,  # Wait for tracking completion
            remote_save=False  # Don't save during testing
        )
    
    def test_basic_completion(self):
        """Test basic completion tracking."""
        response = self.tracked_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        
        assert response is not None
        assert response.choices[0].message.content
        assert response.usage.total_tokens > 0
    
    def test_streaming_completion(self):
        """Test streaming completion tracking."""
        stream = self.tracked_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Count to 3"}],
            stream=True,
            stream_options={"include_usage": True},
            max_tokens=10
        )
        
        chunks = list(stream)
        assert len(chunks) > 0
        
        # Check that usage is included in final chunk
        final_chunk = chunks[-1]
        assert hasattr(final_chunk, 'usage')
        assert final_chunk.usage.total_tokens > 0
```

### Test Callbacks

```python
def test_custom_callback():
    """Test custom response callback."""
    responses = []
    
    def capture_callback(response):
        responses.append(response)
    
    tracked_client = LLMTrackingProxy(
        client,
        provider=Provider.OPENAI,
        response_callback=capture_callback,
        sync_mode=True
    )
    
    response = tracked_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=5
    )
    
    assert len(responses) == 1
    assert responses[0].id == response.id
```

## 📈 Performance Testing

### Load Testing

```python
import time
import concurrent.futures
from llmcosts.tracker import LLMTrackingProxy, Provider

def test_concurrent_requests():
    """Test concurrent request handling."""
    def make_request(i):
        response = tracked_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Request {i}"}],
            max_tokens=5
        )
        return response.id
    
    # Test with multiple concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request, i) for i in range(10)]
        results = [f.result() for f in futures]
    
    assert len(results) == 10
    assert len(set(results)) == 10  # All unique response IDs
```

### Memory Testing

```python
import gc
import psutil
import os

def test_memory_usage():
    """Test memory usage doesn't grow excessively."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Make many requests
    for i in range(100):
        response = tracked_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Request {i}"}],
            max_tokens=5
        )
    
    gc.collect()
    final_memory = process.memory_info().rss
    memory_growth = final_memory - initial_memory
    
    # Memory growth should be reasonable (less than 100MB)
    assert memory_growth < 100 * 1024 * 1024
```

## 🔍 Debugging Tests

### Enable Debug Output

```python
import logging

# Enable debug logging for tests
logging.basicConfig(level=logging.DEBUG)

# Run tests with debug output
pytest tests/test_openai_nonstreaming.py -v -s
```

### Test with Real API Keys

```bash
# Set test environment variables
export LLMCOSTS_API_KEY="your-real-api-key"
export OPENAI_API_KEY="your-real-openai-key"

# Run tests
uv run pytest tests/test_openai_nonstreaming.py -v
```

### Skip Tests Without API Keys

```python
import pytest
import os

@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
def test_openai_integration():
    # Test code here
    pass
```

## 📝 Test Reports

### Generate Coverage Report

```bash
# Generate HTML coverage report
uv run pytest --cov=llmcosts --cov-report=html

# Open coverage report
open htmlcov/index.html
```

### Generate Test Report

```bash
# Generate JUnit XML report
uv run pytest --junitxml=test-results.xml

# Generate detailed test report
uv run pytest --html=test-report.html --self-contained-html
```

## 🔗 Related Documentation

- **[Client Tracking & Context Data](client-tracking.md)** - Track costs per client with rich context data
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions
- **[Configuration](configuration.md)** - Advanced configuration options
- **[Providers](providers.md)** - Provider-specific integration guides 