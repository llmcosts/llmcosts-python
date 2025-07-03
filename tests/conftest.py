"""
Pytest configuration for LLMCosts tests.

This file provides shared fixtures and utilities for running tests with
endpoint integration enabled.
"""

import os

import pytest
from environs import Env

# Load environment variables
env = Env()
env.read_env(os.path.join(os.path.dirname(__file__), ".env"))


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "endpoint: mark test as requiring endpoint integration"
    )
    config.addinivalue_line(
        "markers", "llm_api: mark test as making real LLM API calls"
    )


@pytest.fixture
def endpoint_enabled(request):
    """Fixture that indicates whether endpoint testing is enabled."""
    # Always enabled - no more flag logic
    return True


@pytest.fixture
def llmcosts_api_key():
    """Fixture to get LLMCosts API key."""
    api_key = env.str("LLMCOSTS_API_KEY", None)
    if not api_key:
        pytest.skip("LLMCOSTS_API_KEY not found in environment")
    return api_key


@pytest.fixture(scope="session")
def endpoint_test_summary():
    """Session-scoped fixture to collect endpoint test results."""
    summary = {
        "total_tests": 0,
        "successful_tests": 0,
        "total_requests": 0,
        "total_processed": 0,
        "errors": [],
    }
    yield summary

    # Print summary at the end of the session
    if summary["total_tests"] > 0:
        print("\n=== Endpoint Integration Test Summary ===")
        print(f"Tests run: {summary['total_tests']}")
        print(f"Successful: {summary['successful_tests']}")
        print(f"Total endpoint requests: {summary['total_requests']}")
        print(f"Total usage data processed: {summary['total_processed']}")
        if summary["errors"]:
            print(f"Errors: {len(summary['errors'])}")
            for error in summary["errors"][:5]:  # Show first 5 errors
                print(f"  - {error}")
