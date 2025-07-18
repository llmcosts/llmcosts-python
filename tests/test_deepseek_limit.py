from pathlib import Path
from unittest.mock import patch

import openai
import pytest
from environs import Env

from llmcosts.exceptions import TriggeredLimitError
from llmcosts.tracker import LLMTrackingProxy
from llmcosts.tracker.providers import Provider

env = Env()
env.read_env(Path(__file__).parent / ".env")


@pytest.fixture
def client():
    api_key = env.str("DEEPSEEK_API_KEY", None)
    if not api_key:
        pytest.skip("DEEPSEEK_API_KEY not configured")
    return openai.OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
    )


@pytest.fixture
def tracked_client(client):
    return LLMTrackingProxy(
        client,
        provider=Provider.OPENAI,
        debug=True,
    )


def test_base_url_extraction(client, tracked_client):
    """Test that base_url is automatically extracted from the OpenAI client."""
    # Verify the original client has the correct base_url
    assert client.base_url == "https://api.deepseek.com/v1/"

    # Verify the tracked client auto-extracted the base_url
    assert tracked_client.base_url == "https://api.deepseek.com/v1/"

    # Verify the internal _base_url is set correctly
    assert tracked_client._base_url == "https://api.deepseek.com/v1/"


def _allow():
    return {"status": "checked", "allowed": True, "violations": [], "warnings": []}


def _block():
    violation = {
        "event_id": "ca23a271-7419-48ab-871f-b9eb36a2c73d",
        "threshold_type": "limit",
        "amount": "1.00",
        "period": "daily",
        "triggered_at": "2024-01-01T00:00:00Z",
        "expires_at": "2024-01-02T00:00:00Z",
        "provider": "deepseek",
        "model_id": "deepseek-chat",
        "client_customer_key": None,
        "message": "Usage blocked: limit threshold of $1.00 exceeded",
    }
    return {
        "status": "checked",
        "allowed": False,
        "violations": [violation],
        "warnings": [],
    }


class TestDeepSeekLimit:
    def test_nonstreaming_allowed(self, tracked_client):
        with patch.object(
            tracked_client._llm_costs_client,
            "check_triggered_thresholds",
            return_value=_allow(),
        ):
            res = tracked_client.chat.completions.create(
                model="deepseek-chat",
                max_tokens=10,
                messages=[{"role": "user", "content": "hi"}],
            )
            assert res.choices

    def test_nonstreaming_blocked(self, tracked_client):
        with patch.object(
            tracked_client._llm_costs_client,
            "check_triggered_thresholds",
            return_value=_block(),
        ):
            with pytest.raises(TriggeredLimitError):
                tracked_client.chat.completions.create(
                    model="deepseek-chat",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "hi"}],
                )

    def test_streaming_allowed(self, tracked_client):
        with patch.object(
            tracked_client._llm_costs_client,
            "check_triggered_thresholds",
            return_value=_allow(),
        ):
            stream = tracked_client.chat.completions.create(
                model="deepseek-chat",
                max_tokens=10,
                messages=[{"role": "user", "content": "count"}],
                stream=True,
                stream_options={"include_usage": True},
            )
            chunks = list(stream)
            assert len(chunks) > 0

    def test_streaming_blocked(self, tracked_client):
        with patch.object(
            tracked_client._llm_costs_client,
            "check_triggered_thresholds",
            return_value=_block(),
        ):
            with pytest.raises(TriggeredLimitError):
                list(
                    tracked_client.chat.completions.create(
                        model="deepseek-chat",
                        max_tokens=10,
                        messages=[{"role": "user", "content": "hi"}],
                        stream=True,
                        stream_options={"include_usage": True},
                    )
                )
