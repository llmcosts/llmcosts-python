# Manual Test Scripts

This directory contains manual test scripts that log all LLM interactions to files for detailed review.

## Scripts

### `manual_check_nonstreaming.py`
Tests all non-streaming LLM calls across all providers and models. Logs:
- LLM request data
- LLM response data  
- Usage data sent to tracker
- Server responses from tracker (immediate via sync mode)
- Test results
- Tracker health information

### `manual_check_streaming.py`
Tests all streaming LLM calls across all providers and models. Logs:
- LLM request data
- LLM streaming chunks
- Usage data sent to tracker
- Server responses from tracker (immediate via sync mode)
- Test results
- Tracker health information

## New Tracker Features

The scripts now use the new hybrid tracker functionality:

### Sync Mode
- `sync_mode=True`: Sends usage data immediately and returns server response
- Perfect for manual testing where you need immediate feedback
- Slower than batch mode but provides instant server responses

### Response Access
- `get_last_response()`: Get the most recent server response
- `clear_stored_responses()`: Clear stored responses

### Enhanced Health Monitoring
- `get_health_info()`: Get detailed tracker health information
- Shows thread status, queue size, and operation counts
- Different health info for sync vs async modes
- Helps diagnose tracker performance and issues

### Proper Resource Management
- Automatic cleanup with `shutdown()` method
- Thread-safe operations with proper locking
- Memory-efficient bounded queues
- Graceful handling of thread restarts

### Usage Examples

```python
from llmcosts.tracker.usage_delivery import create_usage_tracker

# For immediate responses (manual testing)
tracker = create_usage_tracker(
    api_key="your_key",
    sync_mode=True
)

# Make LLM call
# ... your LLM calls here ...

# Get the last server response
last_response = tracker.get_last_response()
if last_response:
    print(f"Server processed: {last_response['processed']} records")

# Clean up
tracker.shutdown()
```

## Running the Scripts

From the project root:

```bash
# Non-streaming tests
cd tests/manual
python manual_check_nonstreaming.py

# Streaming tests  
python manual_check_streaming.py
```

Make sure you have your API keys configured in `tests/.env` before running.

## Log Output

Both scripts create timestamped log directories with:
- Individual test result files
- LLM request/response data
- Server response data  
- Summary files with test results and health info

Check the console output for the log directory path and review the individual files for detailed debugging information.

## Log Files

For each model tested, you'll get:
- `{provider}_{model}_llm_request.json` - What was sent to the LLM
- `{provider}_{model}_llm_response.json` - What the LLM returned
- `{provider}_{model}_usage.json` - Usage data captured by tracker
- `{provider}_{model}_server_response.json` - Response from llmcosts.com server
- `{provider}_{model}_test_result.json` - Test success/failure status
- `{provider}_{model}_stream_chunks.json` - Individual streaming chunks (streaming only)

## Key Features

### Immediate Server Responses
- Uses `sync_mode=True` for instant server feedback
- No waiting for batch processing
- Perfect for debugging and manual validation

### Comprehensive Logging
- Every interaction is logged to individual JSON files
- Server responses are captured and logged immediately
- Console output shows real-time progress
- Tracker health information is displayed

### Thread Safety & Resource Management
- Thread-safe operations with proper locking mechanisms
- Automatic cleanup of resources on script completion
- Memory-efficient bounded queues prevent memory leaks
- Graceful handling of thread restarts and failures

### Error Handling
- Graceful handling of missing API keys
- Detailed error logging for debugging
- Continues testing even if some providers fail
- Proper status preservation on shutdown (failed vs stopped)

### Health Monitoring
- Real-time tracker health information
- Thread status and queue monitoring
- Operation counts and performance metrics
- Different health reporting for sync vs async modes

## Notes

- These scripts are NOT picked up by pytest (no `test_` prefix)
- They make real API calls and may incur costs
- All interactions are logged for manual review
- Uses new sync mode for immediate server response capture
- Scripts will skip providers where API keys are not available
- Perfect for validating server responses and usage data mapping
- Automatic cleanup ensures no resource leaks
- Thread-safe design suitable for production environments 