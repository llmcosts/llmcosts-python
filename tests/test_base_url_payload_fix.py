"""
Test for base_url parameter fix in OpenAI handler.

This test specifically verifies that the base_url parameter is correctly
passed to the extract_usage_payload method and included in the usage payload
for non-OpenAI providers using OpenAI-compatible APIs.
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import openai
import pytest
from environs import Env

from llmcosts.tracker import LLMTrackingProxy
from llmcosts.tracker.openai_handler import OpenAIUsageHandler
from llmcosts.tracker.providers import Provider


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI response with usage data."""
    response = Mock()
    response.usage = Mock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 20
    response.usage.total_tokens = 30
    response.usage.model_dump.return_value = {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    }
    response.model = "deepseek-chat"
    response.id = "test-response-id"
    response.choices = [Mock()]
    response.choices[0].message.content = "Hello! I'm doing well, thank you for asking."
    return response


@pytest.fixture
def grok_client():
    """Create a Grok client using OpenAI-compatible interface."""
    env = Env()
    env.read_env(Path(__file__).parent / ".env")

    api_key = env.str("GROK_API_KEY", "sk-test-key")  # Default to test key if not found

    return openai.OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")


@pytest.fixture
def fireworks_client():
    """Create a Fireworks client using OpenAI-compatible interface."""
    env = Env()
    env.read_env(Path(__file__).parent / ".env")

    api_key = env.str(
        "FIREWORKS_API_KEY", "sk-test-key"
    )  # Default to test key if not found

    return openai.OpenAI(
        api_key=api_key, base_url="https://api.fireworks.ai/inference/v1"
    )


@pytest.fixture
def deepseek_client():
    """Create a DeepSeek client using OpenAI-compatible interface."""
    env = Env()
    env.read_env(Path(__file__).parent / ".env")

    api_key = env.str(
        "DEEPSEEK_API_KEY", "sk-test-key"
    )  # Default to test key if not found

    return openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")


class TestBaseUrlParameterFix:
    """Test that the base_url parameter fix works correctly."""

    def test_handler_receives_base_url_parameter(self, mock_openai_response):
        """Test that the OpenAI handler correctly receives and uses the base_url parameter."""
        handler = OpenAIUsageHandler()

        # Test with base_url parameter
        payload = handler.extract_usage_payload(
            mock_openai_response,
            Mock(),
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
        )

        assert payload is not None
        assert "base_url" in payload
        assert payload["base_url"] == "https://api.deepseek.com/v1"
        assert payload["model_id"] == "deepseek-chat"
        assert "usage" in payload

    def test_handler_without_base_url_parameter(self, mock_openai_response):
        """Test that the handler works when no base_url is provided."""
        handler = OpenAIUsageHandler()

        # Test without base_url parameter
        payload = handler.extract_usage_payload(
            mock_openai_response, Mock(), model="gpt-4"
        )

        assert payload is not None
        assert "base_url" not in payload  # Should not be included when None
        assert (
            payload["model_id"] == "deepseek-chat"
        )  # Model comes from response object
        assert "usage" in payload

    def test_handler_with_empty_base_url(self, mock_openai_response):
        """Test that the handler works when base_url is empty string."""
        handler = OpenAIUsageHandler()

        # Test with empty base_url
        payload = handler.extract_usage_payload(
            mock_openai_response, Mock(), base_url="", model="gpt-4"
        )

        assert payload is not None
        assert "base_url" not in payload  # Should not be included when empty
        assert (
            payload["model_id"] == "deepseek-chat"
        )  # Model comes from response object
        assert "usage" in payload

    @patch("llmcosts.tracker.openai_handler.OpenAIUsageHandler.extract_usage_payload")
    def test_proxy_passes_base_url_to_handler(
        self, mock_extract, grok_client, mock_openai_response
    ):
        """Test that LLMTrackingProxy correctly passes base_url to the handler."""
        # Setup mock to return a valid payload
        mock_extract.return_value = {
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            "model_id": "grok-beta",
            "base_url": "https://api.x.ai/v1",
            "response_id": "test-id",
            "timestamp": "2024-01-01T00:00:00Z",
        }

        # Create tracked client
        tracked_client = LLMTrackingProxy(
            grok_client,
            provider=Provider.OPENAI,
            debug=True,
        )

        # Mock the actual API call to avoid real network requests
        mock_create = Mock(return_value=mock_openai_response)
        mock_create.__name__ = "create"  # Add __name__ attribute that proxy expects

        with patch.object(
            tracked_client._target.chat.completions,
            "create",
            mock_create,
        ):
            response = tracked_client.chat.completions.create(
                model="grok-beta",
                messages=[{"role": "user", "content": "Hello"}],
            )

        # Verify extract_usage_payload was called with base_url
        mock_extract.assert_called_once()
        call_args = mock_extract.call_args

        # Check that base_url was passed as a keyword argument
        assert "base_url" in call_args.kwargs
        assert (
            call_args.kwargs["base_url"] == "https://api.x.ai/v1/"
        )  # Note: OpenAI adds trailing slash

    def test_multiple_providers_base_url_extraction(self):
        """Test that base_url is correctly extracted for different providers."""
        test_cases = [
            (
                openai.OpenAI(api_key="test", base_url="https://api.deepseek.com/v1"),
                "https://api.deepseek.com/v1/",
            ),
            (
                openai.OpenAI(api_key="test", base_url="https://api.x.ai/v1"),
                "https://api.x.ai/v1/",
            ),
            (
                openai.OpenAI(
                    api_key="test", base_url="https://api.fireworks.ai/inference/v1"
                ),
                "https://api.fireworks.ai/inference/v1/",
            ),
        ]

        for client, expected_base_url in test_cases:
            tracked_client = LLMTrackingProxy(
                client,
                provider=Provider.OPENAI,
            )

            assert tracked_client.base_url == expected_base_url, (
                f"Expected base_url {expected_base_url}, got {tracked_client.base_url}"
            )

    def test_base_url_included_in_usage_logs(self, deepseek_client, caplog):
        """Test that base_url appears in usage logs when using non-OpenAI providers."""
        if deepseek_client.api_key == "sk-test-key":
            pytest.skip("Skipping live API test - no real API key provided")

        tracked_client = LLMTrackingProxy(
            deepseek_client,
            provider=Provider.OPENAI,
            debug=True,
        )

        try:
            # Make a real API call
            response = tracked_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": "Say 'test' and nothing else"}],
                max_tokens=5,  # Keep it minimal
            )

            # Verify the API call worked
            assert response.choices[0].message.content

            # Check logs for usage tracking
            assert "[LLM costs] OpenAI usage →" in caplog.text

            # Extract the JSON payload from the log
            log_lines = caplog.text.split("\n")
            usage_log_line = None
            for line in log_lines:
                if "[LLM costs] OpenAI usage →" in line:
                    usage_log_line = line
                    break

            assert usage_log_line is not None, "Usage log line not found"

            # Extract JSON from the log line
            json_part = usage_log_line.split("[LLM costs] OpenAI usage → ")[1]
            payload = json.loads(json_part)

            # Print payload for debugging
            print("\n=== USAGE PAYLOAD WITH BASE_URL FIX ===")
            print(json.dumps(payload, indent=2))
            print("=== END PAYLOAD ===\n")

            # Verify base_url is included
            assert "base_url" in payload, "base_url should be included in usage payload"
            assert payload["base_url"] == "https://api.deepseek.com/v1/", (
                f"Expected base_url to be 'https://api.deepseek.com/v1/', "
                f"but got '{payload['base_url']}'"
            )

        except Exception as e:
            if "API key" in str(e) or "authentication" in str(e).lower():
                pytest.skip(f"Skipping live API test - authentication issue: {e}")
            else:
                raise

    def test_streaming_base_url_inclusion(self, deepseek_client, caplog):
        """Test that base_url is included in streaming usage payloads."""
        if deepseek_client.api_key == "sk-test-key":
            pytest.skip("Skipping live API test - no real API key provided")

        tracked_client = LLMTrackingProxy(
            deepseek_client,
            provider=Provider.OPENAI,
            debug=True,
        )

        try:
            # Make a streaming API call
            response_stream = tracked_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": "Say 'hello' and nothing else"}],
                stream=True,
                stream_options={"include_usage": True},
                max_tokens=5,  # Keep it minimal
            )

            # Consume the stream
            full_response = ""
            for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content

            assert full_response.strip()  # Verify we got some content

            # Check logs for usage tracking
            assert "[LLM costs] OpenAI usage →" in caplog.text

            # Find all usage log lines
            log_lines = caplog.text.split("\n")
            usage_log_lines = [
                line for line in log_lines if "[LLM costs] OpenAI usage →" in line
            ]

            # Should have at least one usage log with base_url
            found_base_url = False
            for usage_log_line in usage_log_lines:
                json_part = usage_log_line.split("[LLM costs] OpenAI usage → ")[1]
                payload = json.loads(json_part)

                if "base_url" in payload:
                    found_base_url = True
                    assert payload["base_url"] == "https://api.deepseek.com/v1/"
                    print("\n=== STREAMING USAGE PAYLOAD WITH BASE_URL ===")
                    print(json.dumps(payload, indent=2))
                    print("=== END PAYLOAD ===\n")
                    break

            assert found_base_url, (
                "No usage payload with base_url found in streaming logs"
            )

        except Exception as e:
            if "API key" in str(e) or "authentication" in str(e).lower():
                pytest.skip(f"Skipping live API test - authentication issue: {e}")
            else:
                raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
