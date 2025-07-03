#!/usr/bin/env python3
"""
Manual script to test non-streaming LLM calls and log everything for review.

This script works like the pytest file but logs all interactions to files
for manual review. It tests all providers and models with real API calls.
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


class ManualUsageCapture:
    """Helper class to capture and log all usage data and server responses."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.usage_data: List[Dict[str, Any]] = []
        self.server_responses: List[Dict[str, Any]] = []
        self.errors: List[str] = []
        self.test_results: List[Dict[str, Any]] = []

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

    def log_llm_response(self, provider: str, model: str, response_data: Any):
        """Log LLM response data."""
        # Convert response to dict if it's an object
        if hasattr(response_data, "__dict__"):
            response_dict = response_data.__dict__
        elif hasattr(response_data, "model_dump"):
            response_dict = response_data.model_dump()
        else:
            response_dict = str(response_data)

        log_file = self.log_dir / f"{provider}_{model}_llm_response.json"
        with open(log_file, "w") as f:
            json.dump(response_dict, f, indent=2, default=str)
        print(f"LLM response logged to: {log_file}")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of captured data."""
        return {
            "total_usage_data": len(self.usage_data),
            "total_server_responses": len(self.server_responses),
            "errors": self.errors,
            "test_results": self.test_results,
        }


def setup_usage_capture(log_dir: str):
    """Set up usage capture for the duration of the script."""
    capture = ManualUsageCapture(log_dir)

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
]

GROK_MODELS = [
    "grok-3-mini",
]


def test_openai_models(capture, tracker):
    """Test OpenAI models with manual logging."""
    try:
        import openai
    except ImportError:
        print("OpenAI library not installed, skipping OpenAI tests")
        return

    api_key = env.str("OPENAI_API_KEY", None)
    if not api_key:
        print("OPENAI_API_KEY not found, skipping OpenAI tests")
        return

    from llmcosts.tracker import LLMTrackingProxy

    client = LLMTrackingProxy(openai.OpenAI(api_key=api_key))

    for model in OPENAI_MODELS:
        print(f"\n=== Testing OpenAI {model} ===")
        try:
            # Log the request
            request_data = {
                "model": model,
                "messages": [
                    {"role": "user", "content": "Say hello in exactly 3 words"}
                ],
                "max_tokens": 20,
            }
            capture.log_llm_request("openai", model, request_data)

            # Make the API call
            response = client.chat.completions.create(**request_data)

            # Log the response
            capture.log_llm_response("openai", model, response)

            # Verify the response
            assert response.choices[0].message.content is not None
            assert len(response.choices[0].message.content.strip()) > 0

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
            else:
                print("No tracker available - server response not captured")

            print(f"✅ OpenAI {model}: SUCCESS")
            capture.log_test_result("openai", model, True)

        except Exception as e:
            print(f"❌ OpenAI {model}: FAILED - {e}")
            capture.log_test_result("openai", model, False, str(e))


def test_anthropic_models(capture, tracker):
    """Test Anthropic models with manual logging."""
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
        print(f"\n=== Testing Anthropic {model} ===")
        try:
            # Log the request
            request_data = {
                "model": model,
                "max_tokens": 50,
                "messages": [
                    {"role": "user", "content": "Say hello in exactly 3 words"}
                ],
            }
            capture.log_llm_request("anthropic", model, request_data)

            # Make the API call
            response = client.messages.create(**request_data)

            # Log the response
            capture.log_llm_response("anthropic", model, response)

            # Verify the response
            assert hasattr(response, "content")
            assert len(response.content) > 0
            assert response.content[0].text

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
            else:
                print("No tracker available - server response not captured")

            print(f"✅ Anthropic {model}: SUCCESS")
            capture.log_test_result("anthropic", model, True)

        except Exception as e:
            print(f"❌ Anthropic {model}: FAILED - {e}")
            capture.log_test_result("anthropic", model, False, str(e))


def test_gemini_models(capture, tracker):
    """Test Gemini models with manual logging."""
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
        print(f"\n=== Testing Gemini {model} ===")
        try:
            # Log the request
            request_data = {"model": model, "contents": "Say hello in exactly 3 words"}
            capture.log_llm_request("gemini", model, request_data)

            # Make the API call
            response = client.models.generate_content(**request_data)

            # Log the response
            capture.log_llm_response("gemini", model, response)

            # Verify the response
            assert response.text is not None
            assert len(response.text.strip()) > 0

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
            else:
                print("No tracker available - server response not captured")

            print(f"✅ Gemini {model}: SUCCESS")
            capture.log_test_result("gemini", model, True)

        except Exception as e:
            print(f"❌ Gemini {model}: FAILED - {e}")
            capture.log_test_result("gemini", model, False, str(e))


def test_bedrock_models(capture, tracker):
    """Test Bedrock models with manual logging."""
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
        print(f"\n=== Testing Bedrock {model} ===")
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

            # Make the API call
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
            else:
                print("No tracker available - server response not captured")

            print(f"✅ Bedrock {model}: SUCCESS")
            capture.log_test_result("bedrock", model, True)

        except Exception as e:
            print(f"❌ Bedrock {model}: FAILED - {e}")
            capture.log_test_result("bedrock", model, False, str(e))


def test_deepseek_models(capture, tracker):
    """Test DeepSeek models with manual logging."""
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
        print(f"\n=== Testing DeepSeek {model} ===")
        try:
            # Log the request
            request_data = {
                "model": model,
                "messages": [
                    {"role": "user", "content": "Say hello in exactly 3 words"}
                ],
                "max_tokens": 20,
            }
            capture.log_llm_request("deepseek", model, request_data)

            # Make the API call
            response = client.chat.completions.create(**request_data)

            # Log the response
            capture.log_llm_response("deepseek", model, response)

            # Verify the response
            assert response.choices[0].message.content is not None
            assert len(response.choices[0].message.content.strip()) > 0

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
            else:
                print("No tracker available - server response not captured")

            print(f"✅ DeepSeek {model}: SUCCESS")
            capture.log_test_result("deepseek", model, True)

        except Exception as e:
            print(f"❌ DeepSeek {model}: FAILED - {e}")
            capture.log_test_result("deepseek", model, False, str(e))


def test_grok_models(capture, tracker):
    """Test Grok models with manual logging."""
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
        print(f"\n=== Testing Grok {model} ===")
        try:
            # Log the request
            request_data = {
                "model": model,
                "messages": [
                    {"role": "user", "content": "Say hello in exactly 3 words"}
                ],
                "max_tokens": 20,
            }
            capture.log_llm_request("grok", model, request_data)

            # Make the API call
            response = client.chat.completions.create(**request_data)

            # Log the response
            capture.log_llm_response("grok", model, response)

            # Verify the response
            assert response.choices[0].message.content is not None
            # Grok might return empty content due to content filtering
            if len(response.choices[0].message.content.strip()) == 0:
                print(f"⚠️ Grok {model}: EMPTY CONTENT (content filtering)")
            else:
                print(f"✅ Grok {model}: SUCCESS")

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
            else:
                print("No tracker available - server response not captured")

            capture.log_test_result("grok", model, True)

        except Exception as e:
            print(f"❌ Grok {model}: FAILED - {e}")
            capture.log_test_result("grok", model, False, str(e))


def main():
    """Main function to run all manual tests."""
    print("=== Manual Non-Streaming LLM Usage Tracker Test ===")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Create log directory
    log_dir = (
        Path(__file__).parent
        / "logs"
        / f"nonstreaming_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Log directory: {log_dir}")

    # Set up usage capture and tracker
    capture, tracker = setup_usage_capture(str(log_dir))

    # Test all providers
    test_openai_models(capture, tracker)
    test_anthropic_models(capture, tracker)
    test_gemini_models(capture, tracker)
    test_bedrock_models(capture, tracker)
    test_deepseek_models(capture, tracker)
    test_grok_models(capture, tracker)

    # Get all stored responses
    if tracker:
        print("\n=== Server Response Summary ===")
        last_response = tracker.get_last_response()
        if last_response:
            print(f"Last server response: {json.dumps(last_response, indent=2)}")
        else:
            print("No server response available")
    else:
        print("\n=== Server Response Summary ===")
        print("No tracker available - no server responses captured")

    # Print summary
    summary = capture.get_summary()
    print("\n=== Test Summary ===")
    print(f"Total usage data captured: {summary['total_usage_data']}")
    print(f"Total server responses: {summary['total_server_responses']}")
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
