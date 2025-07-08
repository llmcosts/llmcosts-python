#!/usr/bin/env python3
"""
Test script to verify base_url extraction from OpenAI client.
"""

import openai

from llmcosts.tracker import LLMTrackingProxy, Provider


def test_base_url_extraction():
    """Test that base_url is automatically extracted from OpenAI client."""

    # Test 1: Standard OpenAI client (should extract default base_url)
    print("Test 1: Standard OpenAI client")
    openai_client = openai.OpenAI(api_key="dummy")
    tracked_openai = LLMTrackingProxy(
        openai_client, provider=Provider.OPENAI, api_key="dummy"
    )
    print(f"  Client base_url: {openai_client.base_url}")
    print(f"  Proxy base_url: {tracked_openai.base_url}")
    print(f"  Auto-extracted: {tracked_openai.base_url == openai_client.base_url}")
    print()

    # Test 2: DeepSeek client (should extract custom base_url)
    print("Test 2: DeepSeek client")
    deepseek_client = openai.OpenAI(
        api_key="dummy", base_url="https://api.deepseek.com/v1"
    )
    tracked_deepseek = LLMTrackingProxy(
        deepseek_client, provider=Provider.OPENAI, api_key="dummy"
    )
    print(f"  Client base_url: {deepseek_client.base_url}")
    print(f"  Proxy base_url: {tracked_deepseek.base_url}")
    print(f"  Auto-extracted: {tracked_deepseek.base_url == deepseek_client.base_url}")
    print()

    # Test 3: Explicit base_url should override auto-extraction
    print("Test 3: Explicit base_url override")
    grok_client = openai.OpenAI(api_key="dummy", base_url="https://api.x.ai/v1")
    tracked_grok = LLMTrackingProxy(
        grok_client,
        provider=Provider.OPENAI,
        base_url="https://custom.example.com/v1",  # Explicit override
        api_key="dummy",
    )
    print(f"  Client base_url: {grok_client.base_url}")
    print(f"  Proxy base_url: {tracked_grok.base_url}")
    print(
        f"  Uses explicit: {tracked_grok.base_url == 'https://custom.example.com/v1'}"
    )
    print()


if __name__ == "__main__":
    test_base_url_extraction()
