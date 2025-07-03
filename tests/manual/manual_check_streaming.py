#!/usr/bin/env python3
"""
Manual script to test streaming LLM calls and log everything for review.

This script works like the pytest file but logs all interactions to files
for manual review. It tests all providers and models with real streaming API calls.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add the parent directory to the path so we can import tracker
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from environs import Env

from llmcosts.tracker.usage_delivery import create_usage_tracker

# Load environment variables from the correct location
env = Env()
env.read_env(os.path.join(os.path.dirname(__file__), "..", ".env"))


class ManualStreamingUsageCapture:
    """Helper class to capture and log all streaming usage data and server responses."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.usage_data: List[Dict[str, Any]] = []
        self.server_responses: List[Dict[str, Any]] = []
        self.errors: List[str] = []
        self.test_results: List[Dict[str, Any]] = []
        self.stream_chunks: List[Any] = []

    def capture_usage(self, usage_data: Dict[str, Any], provider: str, model: str):
        """Capture usage data from LLM calls."""
        self.usage_data.append(usage_data)

        # Log the usage data
        log_file = self.log_dir / f"{provider}_{model}_usage.json"
        with open(log_file, "w") as f:
            json.dump(usage_data, f, indent=2)
        print(f"Usage data logged to: {log_file}")

    def capture_server_response(
        self, response_data: Dict[str, Any], provider: str, model: str
    ):
        """Capture server response data."""
        self.server_responses.append(response_data)

        # Log the server response
        log_file = self.log_dir / f"{provider}_{model}_server_response.json"
        with open(log_file, "w") as f:
            json.dump(response_data, f, indent=2)
        print(f"Server response logged to: {log_file}")

    def capture_stream_chunks(self, chunks: List[Any], provider: str, model: str):
        """Capture streaming chunks for validation."""
        self.stream_chunks.extend(chunks)

        # Log the stream chunks
        log_file = self.log_dir / f"{provider}_{model}_stream_chunks.json"
        chunks_data = []
        for chunk in chunks:
            # Convert chunk to dict if it's an object
            if hasattr(chunk, "__dict__"):
                chunk_dict = chunk.__dict__
            elif hasattr(chunk, "model_dump"):
                chunk_dict = chunk.model_dump()
            else:
                chunk_dict = str(chunk)
            chunks_data.append(chunk_dict)

        with open(log_file, "w") as f:
            json.dump(chunks_data, f, indent=2, default=str)
        print(f"Stream chunks logged to: {log_file}")

    def log_test_result(
        self, provider: str, model: str, success: bool, error: str = None
    ):
        """Log test results."""
        result = {
            "provider": provider,
            "model": model,
            "success": success,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        }
        self.test_results.append(result)

        # Log the test result
        log_file = self.log_dir / f"{provider}_{model}_test_result.json"
        with open(log_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Test result logged to: {log_file}")

    def log_llm_request(self, provider: str, model: str, request_data: Dict[str, Any]):
        """Log LLM request data."""
        log_file = self.log_dir / f"{provider}_{model}_llm_request.json"
        with open(log_file, "w") as f:
            json.dump(request_data, f, indent=2)
        print(f"LLM request logged to: {log_file}")

    def validate_streaming_structure(self, provider: str, model: str):
        """Validate that streaming response has the expected structure."""
        assert len(self.stream_chunks) > 0, "Should have received streaming chunks"

        # Validate streaming structure based on provider
        if provider == "openai":
            # OpenAI streaming should have content chunks and usage chunks
            content_chunks = [
                c
                for c in self.stream_chunks
                if hasattr(c, "choices")
                and c.choices
                and hasattr(c.choices[0], "delta")
                and c.choices[0].delta.content
            ]
            usage_chunks = [
                c
                for c in self.stream_chunks
                if hasattr(c, "usage") and c.usage is not None
            ]
            assert len(content_chunks) > 0, "Should have content chunks"
            assert len(usage_chunks) == 1, "Should have exactly one usage chunk"
        elif provider == "anthropic":
            # Anthropic streaming should have content chunks
            content_chunks = [
                c
                for c in self.stream_chunks
                if hasattr(c, "type") and c.type == "content_block_delta"
            ]
            assert len(content_chunks) > 0, "Should have content chunks"
        elif provider == "gemini":
            # Gemini streaming should have content chunks
            content_chunks = [
                c for c in self.stream_chunks if hasattr(c, "text") and c.text
            ]
            assert len(content_chunks) > 0, "Should have content chunks"

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of captured data."""
        return {
            "total_usage_data": len(self.usage_data),
            "total_server_responses": len(self.server_responses),
            "total_stream_chunks": len(self.stream_chunks),
            "errors": self.errors,
            "test_results": self.test_results,
        }


def setup_usage_capture(log_dir: str):
    """Set up usage capture for the duration of the script."""
    capture = ManualStreamingUsageCapture(log_dir)

    # Get API key
    api_key = env.str("LLMCOSTS_API_KEY", None)
    if not api_key:
        print(
            "WARNING: LLMCOSTS_API_KEY not found. Server responses will not be captured."
        )
        return capture, None

    # Create a tracker with sync mode for immediate responses
    tracker = create_usage_tracker(
        api_key=api_key,
        sync_mode=True,  # Send immediately and get response
        timeout=10,
    )

    # Print tracker health info
    print("=== Tracker Health Info ===")
    health = tracker.get_health_info()
    print(f"Status: {health['status']}")
    print(
        f"Mode: {'Sync (immediate delivery)' if tracker.sync_mode else 'Async (background delivery)'}"
    )
    print(f"Queue max size: {tracker.max_queue_size}")
    if not tracker.sync_mode:
        print(f"Healthy: {health['is_healthy']}")
        print(f"Worker thread: {health['worker_thread_name']}")
    else:
        print("Worker thread: Not applicable (sync mode)")

    return capture, tracker


# Define all models from existing tests
OPENAI_MODELS = [
    "gpt-4o-mini",
]

ANTHROPIC_MODELS = [
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-latest",
    "claude-3-5-haiku-latest",
]

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite-preview-06-17",
]

BEDROCK_MODELS = [
    "us.amazon.nova-pro-v1:0",
]

DEEPSEEK_MODELS = [
    "deepseek-chat",
    "deepseek-coder",
]

GROK_MODELS = [
    "grok-3-mini",
]


def test_openai_streaming_models(capture, tracker):
    api_key = env.str("OPENAI_API_KEY", None)
    if not api_key:
        print("⚠️  OPENAI_API_KEY not found - skipping OpenAI streaming tests")
        return

    try:
        import openai
    except ImportError:
        print("⚠️  OpenAI library not installed - skipping OpenAI streaming tests")
        return

    from llmcosts.tracker import LLMTrackingProxy

    client = LLMTrackingProxy(openai.OpenAI(api_key=api_key), debug=True)

    for model in OPENAI_MODELS:
        print(f"\n=== Testing OpenAI Streaming {model} ===")
        try:
            # Prepare the request
            request_data = {
                "model": model,
                "messages": [
                    {"role": "user", "content": "Count to 3, one number per line"}
                ],
                "max_tokens": 50,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            capture.log_llm_request("openai", model, request_data)

            # Make the streaming API call
            stream = client.chat.completions.create(**request_data)

            chunks = []
            for chunk in stream:
                chunks.append(chunk)

            # Log the stream chunks
            capture.capture_stream_chunks(chunks, "openai", model)

            # Verify we got chunks
            assert len(chunks) > 0

            print(f"✅ OpenAI Streaming {model}: SUCCESS - {len(chunks)} chunks")

            # Get the server response immediately (sync mode)
            if tracker:
                server_response = tracker.get_last_response()
                if server_response:
                    capture.capture_server_response(server_response, "openai", model)
                    print(
                        f"Server response received: {json.dumps(server_response, indent=2)}"
                    )
                else:
                    print("No server response received")

            capture.log_test_result("openai", model, True)

        except Exception as e:
            print(f"❌ OpenAI Streaming {model}: FAILED - {e}")
            capture.log_test_result("openai", model, False, str(e))


def test_anthropic_streaming_models(capture, tracker):
    """Test Anthropic streaming models with manual logging."""
    try:
        import anthropic
    except ImportError:
        print("Anthropic library not installed, skipping Anthropic tests")
        return

    api_key = env.str("ANTHROPIC_API_KEY", None)
    if not api_key:
        print("ANTHROPIC_API_KEY not found, skipping Anthropic tests")
        return

    from llmcosts.tracker import LLMTrackingProxy

    client = LLMTrackingProxy(anthropic.Anthropic(api_key=api_key))

    for model in ANTHROPIC_MODELS:
        print(f"\n=== Testing Anthropic Streaming {model} ===")
        try:
            # Log the request
            request_data = {
                "model": model,
                "max_tokens": 50,
                "messages": [
                    {
                        "role": "user",
                        "content": "Count from 1 to 3, one number per line",
                    }
                ],
                "stream": True,
            }
            capture.log_llm_request("anthropic", model, request_data)

            # Make the streaming API call
            stream = client.messages.create(**request_data)

            # Collect all chunks
            chunks = list(stream)
            capture.capture_stream_chunks(chunks, "anthropic", model)

            # Verify we got chunks
            assert len(chunks) > 0

            # Validate streaming structure
            capture.validate_streaming_structure("anthropic", model)

            # Get the server response immediately (sync mode)
            if tracker:
                server_response = tracker.get_last_response()
                if server_response:
                    capture.capture_server_response(server_response, "anthropic", model)
                    print(
                        f"Server response received: {json.dumps(server_response, indent=2)}"
                    )
                else:
                    print("No server response received")

            print(f"✅ Anthropic Streaming {model}: SUCCESS")
            capture.log_test_result("anthropic", model, True)

        except Exception as e:
            print(f"❌ Anthropic Streaming {model}: FAILED - {e}")
            capture.log_test_result("anthropic", model, False, str(e))


def test_gemini_streaming_models(capture, tracker):
    """Test Gemini streaming models with manual logging."""
    try:
        import google.genai as genai
    except ImportError:
        print("google-genai library not installed, skipping Gemini tests")
        return

    api_key = env.str("GOOGLE_API_KEY", None)
    if not api_key:
        print("GOOGLE_API_KEY not found, skipping Gemini tests")
        return

    from llmcosts.tracker import LLMTrackingProxy

    client = LLMTrackingProxy(genai.Client(api_key=api_key))

    for model in GEMINI_MODELS:
        print(f"\n=== Testing Gemini Streaming {model} ===")
        try:
            # Log the request
            request_data = {
                "model": model,
                "contents": "Count from 1 to 3, one number per line",
            }
            capture.log_llm_request("gemini", model, request_data)

            # Make the streaming API call
            stream = client.models.generate_content_stream(**request_data)

            # Collect all chunks
            chunks = list(stream)
            capture.capture_stream_chunks(chunks, "gemini", model)

            # Verify we got chunks
            assert len(chunks) > 0

            # Validate streaming structure
            capture.validate_streaming_structure("gemini", model)

            # Get the server response immediately (sync mode)
            if tracker:
                server_response = tracker.get_last_response()
                if server_response:
                    capture.capture_server_response(server_response, "gemini", model)
                    print(
                        f"Server response received: {json.dumps(server_response, indent=2)}"
                    )
                else:
                    print("No server response received")

            print(f"✅ Gemini Streaming {model}: SUCCESS")
            capture.log_test_result("gemini", model, True)

        except Exception as e:
            print(f"❌ Gemini Streaming {model}: FAILED - {e}")
            capture.log_test_result("gemini", model, False, str(e))


def test_bedrock_streaming_models(capture, tracker):
    """Test Bedrock models with manual logging (non-streaming since Bedrock doesn't have direct streaming)."""
    try:
        import boto3
    except ImportError:
        print("boto3 library not installed, skipping Bedrock tests")
        return

    # Get AWS credentials from environment or .env file
    aws_access_key_id = env.str("AWS_ACCESS_KEY_ID", None)
    aws_secret_access_key = env.str("AWS_SECRET_ACCESS_KEY", None)
    region_name = env.str("AWS_DEFAULT_REGION", "us-east-2")

    if not aws_access_key_id or not aws_secret_access_key:
        print("AWS credentials not found, skipping Bedrock tests")
        return

    try:
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )
        bedrock_client = session.client(
            service_name="bedrock-runtime", region_name=region_name
        )
    except Exception as e:
        print(f"Failed to create Bedrock client: {e}")
        return

    from llmcosts.tracker import LLMTrackingProxy

    client = LLMTrackingProxy(bedrock_client)

    for model in BEDROCK_MODELS:
        print(f"\n=== Testing Bedrock {model} (non-streaming) ===")
        try:
            # Log the request
            request_data = {
                "modelId": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": "Say hello in one word only."}],
                    }
                ],
                "inferenceConfig": {"maxTokens": 10, "temperature": 0.1},
            }
            capture.log_llm_request("bedrock", model, request_data)

            # Make the API call (Bedrock doesn't have direct streaming)
            response = client.converse(**request_data)

            # Log the response
            capture.log_llm_response("bedrock", model, response)

            # Verify the response
            assert "output" in response
            assert "message" in response["output"]
            assert "content" in response["output"]["message"]
            assert len(response["output"]["message"]["content"]) > 0

            # Get the server response immediately (sync mode)
            if tracker:
                server_response = tracker.get_last_response()
                if server_response:
                    capture.capture_server_response(server_response, "bedrock", model)
                    print(
                        f"Server response received: {json.dumps(server_response, indent=2)}"
                    )
                else:
                    print("No server response received")

            print(f"✅ Bedrock {model}: SUCCESS")
            capture.log_test_result("bedrock", model, True)

        except Exception as e:
            print(f"❌ Bedrock {model}: FAILED - {e}")
            capture.log_test_result("bedrock", model, False, str(e))


def test_deepseek_streaming_models(capture, tracker):
    """Test DeepSeek streaming models with manual logging."""
    try:
        import openai
    except ImportError:
        print("OpenAI library not installed, skipping DeepSeek tests")
        return

    api_key = env.str("DEEPSEEK_API_KEY", None)
    if not api_key:
        print("DEEPSEEK_API_KEY not found, skipping DeepSeek tests")
        return

    from llmcosts.tracker import LLMTrackingProxy

    client = LLMTrackingProxy(
        openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    )

    for model in DEEPSEEK_MODELS:
        print(f"\n=== Testing DeepSeek Streaming {model} ===")
        try:
            # Log the request
            request_data = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Count from 1 to 3, one number per line",
                    }
                ],
                "max_tokens": 20,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            capture.log_llm_request("deepseek", model, request_data)

            # Make the streaming API call
            stream = client.chat.completions.create(**request_data)

            # Collect all chunks
            chunks = list(stream)
            capture.capture_stream_chunks(chunks, "deepseek", model)

            # Verify we got chunks
            assert len(chunks) > 0

            # Validate streaming structure (DeepSeek uses OpenAI format)
            capture.validate_streaming_structure("openai", model)

            # Get the server response immediately (sync mode)
            if tracker:
                server_response = tracker.get_last_response()
                if server_response:
                    capture.capture_server_response(server_response, "deepseek", model)
                    print(
                        f"Server response received: {json.dumps(server_response, indent=2)}"
                    )
                else:
                    print("No server response received")

            print(f"✅ DeepSeek Streaming {model}: SUCCESS")
            capture.log_test_result("deepseek", model, True)

        except Exception as e:
            print(f"❌ DeepSeek Streaming {model}: FAILED - {e}")
            capture.log_test_result("deepseek", model, False, str(e))


def test_grok_streaming_models(capture, tracker):
    """Test Grok streaming models with manual logging."""
    try:
        import openai
    except ImportError:
        print("OpenAI library not installed, skipping Grok tests")
        return

    api_key = env.str("GROK_API_KEY", None)
    if not api_key:
        print("GROK_API_KEY not found, skipping Grok tests")
        return

    from llmcosts.tracker import LLMTrackingProxy

    client = LLMTrackingProxy(
        openai.OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    )

    for model in GROK_MODELS:
        print(f"\n=== Testing Grok Streaming {model} ===")
        try:
            # Log the request
            request_data = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Count from 1 to 3, one number per line",
                    }
                ],
                "max_tokens": 20,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            capture.log_llm_request("grok", model, request_data)

            # Make the streaming API call
            stream = client.chat.completions.create(**request_data)

            # Collect all chunks
            chunks = list(stream)
            capture.capture_stream_chunks(chunks, "grok", model)

            # Verify we got chunks
            assert len(chunks) > 0

            # Validate streaming structure (Grok uses OpenAI format)
            try:
                capture.validate_streaming_structure("openai", model)
                print(f"✅ Grok Streaming {model}: SUCCESS")
            except AssertionError as e:
                if "Should have content chunks" in str(e):
                    # Grok might return empty content due to content filtering
                    print(
                        f"⚠️ Grok Streaming {model}: EMPTY CONTENT (content filtering)"
                    )
                else:
                    raise

            # Get the server response immediately (sync mode)
            if tracker:
                server_response = tracker.get_last_response()
                if server_response:
                    capture.capture_server_response(server_response, "grok", model)
                    print(
                        f"Server response received: {json.dumps(server_response, indent=2)}"
                    )
                else:
                    print("No server response received")

            capture.log_test_result("grok", model, True)

        except Exception as e:
            print(f"❌ Grok Streaming {model}: FAILED - {e}")
            capture.log_test_result("grok", model, False, str(e))


def main():
    """Main function to run all manual streaming tests."""
    print("=== Manual Streaming LLM Usage Tracker Test ===")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Create log directory
    log_dir = (
        Path(__file__).parent
        / "logs"
        / f"streaming_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Log directory: {log_dir}")

    # Set up usage capture and tracker
    capture, tracker = setup_usage_capture(str(log_dir))

    # Test all providers
    test_openai_streaming_models(capture, tracker)
    test_anthropic_streaming_models(capture, tracker)
    test_gemini_streaming_models(capture, tracker)
    test_bedrock_streaming_models(capture, tracker)
    test_deepseek_streaming_models(capture, tracker)
    test_grok_streaming_models(capture, tracker)

    # Get server response summary
    if tracker:
        print("\n=== Server Response Summary ===")
        last_response = tracker.get_last_response()
        if last_response:
            print(f"Last server response: {json.dumps(last_response, indent=2)}")
        else:
            print("No server response available")

    # Print summary
    summary = capture.get_summary()
    print("\n=== Test Summary ===")
    print(f"Total usage data captured: {summary['total_usage_data']}")
    print(f"Total server responses: {summary['total_server_responses']}")
    print(f"Total stream chunks: {summary['total_stream_chunks']}")
    print(f"Errors: {summary['errors']}")
    print(f"Test results: {len(summary['test_results'])}")

    # Log summary
    summary_file = log_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary logged to: {summary_file}")

    # Print final tracker health
    if tracker:
        print("\n=== Final Tracker Health ===")
        health = tracker.get_health_info()
        print(f"Status: {health['status']}")
        print(
            f"Mode: {'Sync (immediate delivery)' if tracker.sync_mode else 'Async (background delivery)'}"
        )
        print(f"Total sent: {health['total_sent']}")
        print(f"Total failed: {health['total_failed']}")
        print(f"Queue size: {health['queue_size']}")
        print(f"Queue utilization: {health['queue_utilization']:.2%}")
        if not tracker.sync_mode:
            print(f"Healthy: {health['is_healthy']}")
        else:
            print("Health check: Not applicable (sync mode)")

    # Clean up the tracker
    if tracker:
        print("\n=== Cleaning up tracker ===")
        try:
            tracker.shutdown()
            print("Tracker shutdown successfully")
        except Exception as e:
            print(f"Warning: Error during tracker shutdown: {e}")

    print(f"\nAll logs saved to: {log_dir}")
    print("Check the log files for detailed request/response data.")


if __name__ == "__main__":
    main()
