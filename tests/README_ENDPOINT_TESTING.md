# Endpoint Integration Testing

This directory contains comprehensive tests for validating the LLMCosts endpoint integration across all supported LLM providers. The tests ensure that usage data is correctly sent to the `https://llmcosts.com/api/v1/usage` endpoint and that responses match the expected format.

## The Unified Testing Tool: `check.py`

The `check.py` script is your primary tool for both automated testing and manual validation. It operates in two distinct modes:

### 🤖 **Automated Testing Mode** (`--test` flag)
Run comprehensive test suites with pytest integration and flexible filtering options.

### 🧪 **Manual Testing Mode** (default)
Perform immediate API calls to individual providers with real-time endpoint validation and cost tracking.

## Quick Start Examples

```bash
# Manual testing - test a specific provider/model immediately
uv run python tests/check.py openai gpt-4o-mini
uv run python tests/check.py anthropic claude-3-5-haiku-20241022 --stream

# Automated testing - run comprehensive test suites  
uv run python tests/check.py --test                              # All tests
uv run python tests/check.py --test --test-provider openai       # OpenAI only
uv run python tests/check.py --test --infrastructure             # Infrastructure tests
```

## Test Structure Overview

The test suite is organized into three main categories:

### 1. Provider-Specific Tests
Tests for individual LLM providers, separated by streaming and non-streaming modes:

**OpenAI Tests:**
- `test_openai_nonstreaming.py` - Chat completions, legacy completions, responses API
- `test_openai_streaming.py` - Streaming chat completions with usage tracking

**Anthropic Tests:**
- `test_anthropic_nonstreaming.py` - Claude messages API with comprehensive validation
- `test_anthropic_streaming.py` - Streaming Claude messages

**Google Gemini Tests:**
- `test_gemini_nonstreaming.py` - Gemini generation API
- `test_gemini_streaming.py` - Streaming Gemini generation

**Amazon Bedrock Tests:**
- `test_bedrock_nonstreaming.py` - Bedrock converse API with cost event validation
- `test_bedrock_streaming.py` - Streaming Bedrock API calls

**DeepSeek Tests:**
- `test_deepseek_nonstreaming.py` - DeepSeek API (OpenAI-compatible)
- `test_deepseek_streaming.py` - Streaming DeepSeek API calls

**Grok Tests:**
- `test_grok_nonstreaming.py` - xAI Grok API (OpenAI-compatible)
- `test_grok_streaming.py` - Streaming Grok API calls

**Response Callback Tests:**
- `test_sqlite_callback.py` - SQLite database callback functionality and data validation
- `test_text_callback.py` - JSON Lines text file callback functionality and data validation

### 2. Infrastructure Tests
Core endpoint and framework functionality tests:

- **`test_endpoint_integration.py`** - Mock and real endpoint testing, response format validation
- **`test_endpoint_with_real_llms.py`** - Real LLM API calls with endpoint delivery validation
- **`test_save_and_cost_events.py`** - Cost event saving, retrieval, and `remote_save` flag testing
- **`test_proxy_new_features.py`** - Proxy features: `remote_save`, `context`, `response_callback`
- **`test_usage_tracker.py`** - Usage tracker functionality, batching, error handling
- **`test_logging.py`** - Debug logging functionality
- **`test_thread_safety.py`** - Thread safety and resource management

### 3. Manual Testing Utilities
Interactive testing scripts for comprehensive validation:

- **`check.py`** - Unified tool for both automated and manual testing
- **`manual/manual_check_nonstreaming.py`** - Interactive non-streaming tests across all providers
- **`manual/manual_check_streaming.py`** - Interactive streaming tests across all providers
- **`manual/README.md`** - Manual testing documentation

## Expected Response Format

The endpoint returns responses in this format:

```json
{
  "status": "success",
  "processed": 1,
  "failed": null,
  "errors": null,
  "timestamp": "2025-06-23T17:17:27.881473",
  "events": [
    {
      "model_id": "gpt-4o-mini",
      "provider": "openai",
      "response_id": "chatcmpl-123",
      "input_tokens": 10,
      "output_tokens": 5,
      "total_cost": 0.00015,
      "input_cost": 0.00010,
      "output_cost": 0.00005,
      "timestamp": "2025-06-23T17:17:27.881473",
      "context": null
    }
  ]
}
```

Where:
- `status`: "success", "partial_success", or "failed"
- `processed`: Integer count of usage segments processed successfully
- `failed`: Integer count of failed processing attempts (can be null)
- `errors`: Array of error messages (can be null)
- `events`: Array of cost events with calculated costs and token counts
- `timestamp`: ISO format timestamp of when the request was processed

## Setup

1. **Copy the environment template:**
   ```bash
   cp tests/env.example tests/.env
   ```

2. **Add your API keys to `tests/.env`:**
   ```bash
   # Required for endpoint testing
   LLMCOSTS_API_KEY=your_llmcosts_api_key_here
   
   # Required for LLM provider tests
   OPENAI_API_KEY=your_openai_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   AWS_ACCESS_KEY_ID=your_aws_access_key_id
   AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
   AWS_DEFAULT_REGION=us-east-2
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   XAI_API_KEY=your_xai_grok_api_key_here
   
   # Optional: For callback testing (if not set, temporary directories are used)
   SQLITE_CALLBACK_TARGET_PATH=./test_data
   TEXT_CALLBACK_TARGET_PATH=./test_logs
   ```

3. **Verify setup with a quick test:**
   ```bash
   uv run python tests/check.py --help
   ```

## Using the Unified Testing Tool

### 🧪 Manual Testing Mode (Quick Validation)

Perfect for immediate testing and debugging individual providers:

```bash
# Basic usage: test provider and model
uv run python tests/check.py <provider> <model> [--stream]

# Examples:
uv run python tests/check.py openai gpt-4o-mini                          # Non-streaming
uv run python tests/check.py openai gpt-4o-mini --stream                 # Streaming
uv run python tests/check.py anthropic claude-3-5-haiku-20241022         # Claude non-streaming
uv run python tests/check.py anthropic claude-3-5-haiku-20241022 --stream # Claude streaming
uv run python tests/check.py bedrock us.amazon.nova-pro-v1:0             # Bedrock
uv run python tests/check.py gemini gemini-1.5-flash --stream            # Gemini streaming
uv run python tests/check.py deepseek deepseek-chat                      # DeepSeek
uv run python tests/check.py grok grok-beta --stream                     # Grok streaming
```

**Manual testing provides:**
- 🎯 Immediate API call execution
- 📊 Real-time endpoint response validation
- 💰 Cost event display with calculated costs
- ✅ Response structure validation
- 🐛 Detailed error reporting and troubleshooting info

**Sample Output:**
```
🧪 Manual testing: openai / gpt-4o-mini (non-streaming)
============================================================
Using LLMCosts API key: sk-proj-...
📞 Making API call to openai...
Non-streaming openai chat completion...
✅ Response received: Hello! How can I help you today?

📊 Getting endpoint response...
--- LLMCosts Endpoint Response ---
{
  "status": "success",
  "processed": 1,
  "events": [{
    "model_id": "gpt-4o-mini",
    "provider": "openai", 
    "total_cost": 0.000015,
    "input_tokens": 8,
    "output_tokens": 6
  }]
}
✅ Endpoint response successful
✅ Cost events received: 1
💰 Total cost: $0.000015
✅ Manual test completed!
```

### 🤖 Automated Testing Mode (Comprehensive Test Suites)

For running complete test suites with pytest integration:

```bash
# Basic automated testing
uv run python tests/check.py --test [options]

# Run all provider tests (both streaming and non-streaming)
uv run python tests/check.py --test

# Run tests for specific providers
uv run python tests/check.py --test --test-provider openai
uv run python tests/check.py --test --test-provider anthropic
uv run python tests/check.py --test --test-provider bedrock

# Filter by test type
uv run python tests/check.py --test --streaming-only              # Only streaming tests
uv run python tests/check.py --test --nonstreaming-only           # Only non-streaming tests
uv run python tests/check.py --test --infrastructure              # Only infrastructure tests

# Run specific test files (including callback tests)
uv run python tests/check.py --test tests/test_sqlite_callback.py
uv run python tests/check.py --test tests/test_text_callback.py

# Combine filters
uv run python tests/check.py --test --test-provider openai --streaming-only
uv run python tests/check.py --test --test-provider gemini --nonstreaming-only

# Debugging and development
uv run python tests/check.py --test --collect-only                # Just show what would run
uv run python tests/check.py --test --test-provider deepseek --verbose --capture
uv run python tests/check.py --test tests/test_openai_streaming.py  # Specific files
```

**Supported Providers:** `openai`, `anthropic`, `gemini`, `bedrock`, `deepseek`, `grok`

### Alternative: Direct pytest Usage

You can also run tests directly with pytest if preferred:

```bash
# Run all tests
uv run pytest tests/

# Run specific test categories
uv run pytest tests/test_endpoint_integration.py                   # Endpoint tests
uv run pytest tests/test_openai_*.py                               # All OpenAI tests
uv run pytest tests/test_*_streaming.py                            # All streaming tests

# Run with specific options
uv run pytest -v tests/test_anthropic_nonstreaming.py             # Verbose output
uv run pytest --collect-only tests/                               # Just collect tests
```

## Key Test Features

### Endpoint Response Validation
All tests validate that endpoint responses contain:
- Correct `status` field ("success", "partial_success", or "failed")
- Valid `processed` count matching batch size
- Proper `timestamp` in ISO format
- Cost events with calculated token costs when available

### Usage Data Structure Validation
Tests verify that usage data sent to the endpoint includes:
- `model_id`: The LLM model identifier
- `provider`: The LLM provider name
- `usage`: Token counts and usage metrics
- `response_id`: Unique identifier for the response
- `timestamp`: ISO format timestamp when the usage occurred
- `context`: Optional context data (user_id, session_id, etc.)

### Cost Event Validation
Tests specifically validate cost events returned by the endpoint:
- Token counts (`input_tokens`, `output_tokens`)
- Cost calculations (`input_cost`, `output_cost`, `total_cost`)
- Model and provider information
- Response correlation via `response_id`

### Proxy Feature Testing
Tests cover advanced proxy features:
- **`remote_save`**: Control whether events are saved to database
- **`context`**: Add custom context data to usage tracking
- **`response_callback`**: Execute callbacks after LLM responses
- **Thread safety**: Concurrent usage and resource management

### Error Handling
Tests verify robust error handling for:
- Network connectivity issues
- Authentication failures (401 errors)
- Validation errors (422 responses)
- Server errors (5xx responses)
- Batch processing failures

## Provider-Specific Testing Details

### OpenAI
- **Non-streaming**: Chat completions, legacy completions, responses API
- **Streaming**: Chat completions with `stream=True` and `include_usage=True`
- **Models tested**: `gpt-4o-mini`, `gpt-3.5-turbo-instruct`
- **Manual testing**: `uv run python tests/check.py openai gpt-4o-mini [--stream]`

### Anthropic
- **Non-streaming**: Messages API with comprehensive payload validation
- **Streaming**: Messages API with `stream=True`
- **Models tested**: `claude-sonnet-4-20250514`, `claude-3-5-haiku-20241022`
- **Manual testing**: `uv run python tests/check.py anthropic claude-3-5-haiku-20241022 [--stream]`

### Google Gemini
- **Non-streaming**: `generate_content` API
- **Streaming**: `generate_content` with streaming
- **Models tested**: `gemini-1.5-flash`, `gemini-1.5-pro`
- **Manual testing**: `uv run python tests/check.py gemini gemini-1.5-flash [--stream]`

### Amazon Bedrock
- **Non-streaming**: Converse API with cost event validation
- **Streaming**: Converse API with streaming
- **Models tested**: `us.meta.llama3-3-70b-instruct-v1:0`, `us.amazon.nova-pro-v1:0`
- **Manual testing**: `uv run python tests/check.py bedrock us.amazon.nova-pro-v1:0 [--stream]`

### DeepSeek & Grok
- **Both providers**: OpenAI-compatible API testing
- **Models tested**: Latest available models for each provider
- **Manual testing**: 
  - `uv run python tests/check.py deepseek deepseek-chat [--stream]`
  - `uv run python tests/check.py grok grok-beta [--stream]`

## Response Callback Testing

The test suite includes comprehensive validation of the built-in response callbacks for recording LLM cost events to SQLite databases and JSON Lines text files.

### SQLite Callback Tests (`test_sqlite_callback.py`)

**What is tested:**
- ✅ Real OpenAI API calls with callback integration
- ✅ Database and table creation (automatic schema setup)
- ✅ Cost event data extraction and storage
- ✅ Record overwriting for duplicate `response_id` values
- ✅ Environment variable configuration validation
- ✅ Error handling with malformed response objects
- ✅ Complex context data with nested structures and unicode
- ✅ Model name normalization (e.g., `gpt-4o-mini-2024-07-18` → `gpt-4o-mini`)

**Environment setup required:**
```bash
# Add to tests/.env
SQLITE_CALLBACK_TARGET_PATH=./test_data
```

**Database structure created:**
```sql
CREATE TABLE cost_events (
    response_id TEXT PRIMARY KEY,
    model_id TEXT,
    provider TEXT,
    timestamp TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    context TEXT,
    created_at TEXT,
    updated_at TEXT
)
```

**Running SQLite callback tests:**
```bash
# Run all SQLite callback tests
uv run pytest tests/test_sqlite_callback.py -v

# Run with real API calls
uv run pytest tests/test_sqlite_callback.py::test_sqlite_callback_with_real_openai_streaming -v

# Test environment configuration
uv run pytest tests/test_sqlite_callback.py::test_environment_variable_missing -v
```

### Text File Callback Tests (`test_text_callback.py`)

**What is tested:**
- ✅ Real OpenAI API calls with text file recording
- ✅ JSON Lines format creation and validation
- ✅ Cost event data extraction and serialization
- ✅ Record overwriting with timestamp preservation
- ✅ Environment variable configuration validation
- ✅ Error handling with malformed response objects
- ✅ Complex context data with nested JSON structures
- ✅ Model name normalization and data consistency

**Environment setup required:**
```bash
# Add to tests/.env  
TEXT_CALLBACK_TARGET_PATH=./test_logs
```

**File format created (JSON Lines):**
```json
{"response_id": "chatcmpl-123", "model_id": "gpt-4o-mini", "provider": "openai", "timestamp": "2024-01-15T10:30:00Z", "input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "context": {"user_id": "test"}, "created_at": "2024-01-15T10:30:00Z", "updated_at": "2024-01-15T10:30:00Z"}
```

**Running text callback tests:**
```bash
# Run all text callback tests
uv run pytest tests/test_text_callback.py -v

# Test with real API calls
uv run pytest tests/test_text_callback.py::test_text_callback_with_real_openai_nonstreaming -v

# Test record overwriting behavior
uv run pytest tests/test_text_callback.py::test_text_callback_overwrite_behavior -v
```

### Key Callback Test Features

**Real API Integration:** Both test suites make actual OpenAI API calls to validate end-to-end functionality with real LLM responses.

**Data Validation:** Tests verify that extracted cost event data matches this structure:
```json
{
  "response_id": "chatcmpl-xyz...",
  "model_id": "gpt-4o-mini",
  "provider": "openai", 
  "timestamp": "2025-01-15T10:30:00+00:00",
  "input_tokens": 14,
  "output_tokens": 3,
  "total_tokens": 17,
  "context": {"user_id": "test", "session": "123"},
  "created_at": "2025-01-15T10:30:00+00:00",
  "updated_at": "2025-01-15T10:30:00+00:00"
}
```

**Error Resilience:** Tests ensure callbacks handle malformed responses gracefully without interrupting main API calls.

**Unicode Support:** Both callbacks are tested with complex unicode characters in context data to ensure international character support.

**Cleanup:** All tests use `tempfile.TemporaryDirectory()` for automatic cleanup of test databases and files.

### Environment Configuration for Callback Tests

Set up callback test environment variables in `tests/.env`:

```bash
# SQLite callback test configuration
SQLITE_CALLBACK_TARGET_PATH=./test_data

# Text callback test configuration  
TEXT_CALLBACK_TARGET_PATH=./test_logs

# Required for real API calls in callback tests
OPENAI_API_KEY=your_openai_api_key_here
```

### Running All Callback Tests

```bash
# Run all callback tests together
uv run pytest tests/test_sqlite_callback.py tests/test_text_callback.py -v

# Run with coverage reporting
uv run pytest tests/test_*_callback.py --cov=tracker.callbacks --cov-report=html

# Run only callback tests that use real APIs
uv run pytest tests/test_*_callback.py -k "real_openai" -v
```

**Note:** Callback tests automatically skip if required API keys are missing, but will test all error handling and configuration validation regardless.

## Debugging and Troubleshooting

### 1. Quick Manual Validation
Start with manual testing to isolate issues:

```bash
# Test basic connectivity and authentication
uv run python tests/check.py openai gpt-4o-mini

# Test streaming functionality
uv run python tests/check.py openai gpt-4o-mini --stream

# Test other providers
uv run python tests/check.py anthropic claude-3-5-haiku-20241022
```

### 2. Verbose Automated Testing
Run automated tests with maximum verbosity:

```bash
# Full verbose output for a specific provider
uv run python tests/check.py --test --test-provider openai --verbose --capture

# Test collection only (to see what would run)
uv run python tests/check.py --test --collect-only --test-provider openai
```

### 3. Detailed Test Debugging
Run individual tests with pytest for detailed output:

```bash
# Single test with maximum verbosity
uv run pytest -vvv -s tests/test_endpoint_integration.py::TestRealEndpointValidation::test_real_endpoint_response_format

# Specific provider test with debug info
uv run pytest -vvv -s tests/test_openai_nonstreaming.py::TestOpenAINonStreaming::test_chat_completions_non_streaming
```

### 4. Environment and Configuration Issues
Check your setup:

```bash
# Verify environment file exists
ls tests/.env

# Test the help system
uv run python tests/check.py --help

# Check if all dependencies are installed
uv run python -c "import openai, anthropic; print('Dependencies OK')"
```

### 5. API Key and Authentication Issues
Common authentication problems:

- **Missing API keys**: Check that all required keys are in `tests/.env`
- **Invalid keys**: Manual testing will show authentication errors immediately
- **Network issues**: Manual testing shows connection problems clearly

## Configuration Files

### `check.py`
The unified testing tool with dual modes:
- **Manual mode**: Direct provider/model testing with immediate feedback
- **Automated mode**: Full pytest integration with flexible filtering

### `conftest.py`
Provides shared pytest fixtures:
- `endpoint_enabled`: Always True (endpoint testing is standard)
- `llmcosts_api_key`: LLMCosts API key fixture
- `endpoint_test_summary`: Session-scoped result collection

### `env.example`
Template for environment variables with examples for all required API keys.

## Adding New Provider Tests

To add tests for a new provider:

1. **Create provider-specific test files:**
   ```bash
   tests/test_newprovider_nonstreaming.py
   tests/test_newprovider_streaming.py
   ```

2. **Follow the established pattern:**
   - Import the provider's client library
   - Create fixtures for client and tracked client
   - Test both streaming and non-streaming modes
   - Validate usage data structure and endpoint responses

3. **Update the unified test tool:**
   - Add the new provider to the choices in `check.py` (both manual and automated modes)
   - Update the client building logic in `build_client()` function

4. **Add environment variables:**
   - Update `env.example` with required API keys
   - Document setup in this README

5. **Test the integration:**
   ```bash
   # Test manual mode
   uv run python tests/check.py newprovider some-model

   # Test automated mode
   uv run python tests/check.py --test --test-provider newprovider
   ```

## Advanced Testing Scenarios

### Comprehensive Manual Testing
For systematic manual validation across providers:

```bash
# Test all major providers manually
uv run python tests/manual/manual_check_nonstreaming.py
uv run python tests/manual/manual_check_streaming.py
```

### Infrastructure-Only Testing
Focus on core functionality without provider API calls:

```bash
uv run python tests/check.py --test --infrastructure
```

### Performance and Load Testing
Test thread safety and concurrent usage:

```bash
uv run python tests/check.py --test tests/test_thread_safety.py
```

### Cost Event Validation
Specifically test cost tracking and event generation:

```bash
uv run python tests/check.py --test tests/test_save_and_cost_events.py
```

---

## Summary

The unified `check.py` tool provides a comprehensive testing solution that covers:

- ✅ **Immediate validation** with manual testing mode
- ✅ **Comprehensive coverage** with automated testing mode  
- ✅ **All 6 LLM providers** with both streaming and non-streaming support
- ✅ **Real endpoint integration** with cost event validation
- ✅ **Developer-friendly** with clear output and debugging features
- ✅ **CI/CD ready** with flexible filtering and pytest integration

This testing framework ensures that the LLMCosts endpoint integration works correctly across all supported LLM providers and usage scenarios, providing reliable cost tracking and usage analytics. 