"""Tests for the new framework parameter in LLMTrackingProxy."""

from unittest.mock import MagicMock

import pytest

from llmcosts.tracker import LLMTrackingProxy
from llmcosts.tracker.providers import Provider
from llmcosts.tracker.frameworks import Framework


class MockClient:
    def __init__(self):
        self.completions = MagicMock()


class TestFrameworkParameter:
    def test_framework_enables_langchain_mode(self):
        client = MockClient()
        proxy = LLMTrackingProxy(client, provider=Provider.OPENAI, framework=Framework.LANGCHAIN)
        assert proxy._langchain_mode is True
        # Child proxies inherit
        child = proxy.completions
        assert isinstance(child, LLMTrackingProxy)
        assert child._langchain_mode is True

    def test_framework_default_none(self):
        client = MockClient()
        proxy = LLMTrackingProxy(client, provider=Provider.OPENAI)
        assert proxy._langchain_mode is False
        assert proxy.framework is None

