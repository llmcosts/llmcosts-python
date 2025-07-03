#!/usr/bin/env python3
"""
Comprehensive LLMCosts testing and validation tool.

This script provides both automated test running and manual testing capabilities
for validating the LLMCosts provider integration and endpoint delivery.

AUTOMATED TEST RUNNING:
    # Run all provider tests (both streaming and non-streaming)
    python tests/check.py --test

    # Run only infrastructure tests
    python tests/check.py --test --infrastructure

    # Run tests for a specific provider
    python tests/check.py --test --test-provider openai
    python tests/check.py --test --test-provider bedrock

    # Run only streaming tests (all providers)
    python tests/check.py --test --streaming-only

    # Run only non-streaming tests (all providers)
    python tests/check.py --test --nonstreaming-only

    # Combine provider and streaming mode
    python tests/check.py --test --test-provider openai --streaming-only
    python tests/check.py --test --test-provider anthropic --nonstreaming-only

    # Just collect tests without running them
    python tests/check.py --test --collect-only

    # Verbose output for debugging
    python tests/check.py --test --test-provider deepseek --verbose

    # Run specific test files directly
    python tests/check.py --test tests/test_openai_streaming.py tests/test_bedrock_nonstreaming.py

MANUAL TESTING:
    # Test a specific provider and model manually
    python tests/check.py openai gpt-4o-mini
    python tests/check.py anthropic claude-3-5-haiku-20241022 --stream
    python tests/check.py bedrock us.amazon.nova-pro-v1:0
    python tests/check.py gemini gemini-1.5-flash --stream

Supported Providers:
    - openai: OpenAI GPT models (chat completions, legacy completions, responses API)
    - anthropic: Anthropic Claude models (messages API)
    - gemini: Google Gemini models (generate_content API)
    - bedrock: Amazon Bedrock models (converse API)
    - deepseek: DeepSeek models (OpenAI-compatible API)
    - grok: xAI Grok models (OpenAI-compatible API)
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from environs import Env

from llmcosts.tracker import LLMTrackingProxy
from llmcosts.tracker.providers import Provider

# Load environment variables from .env file in the tests directory
env = Env()
env.read_env(os.path.join(os.path.dirname(__file__), ".env"))


def build_client(provider: str):
    """Create a real LLM client for manual testing."""
    if provider == "openai":
        import openai

        api_key = env.str("OPENAI_API_KEY", None)
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable not set. "
                "Please copy env.example to tests/.env and add your API keys."
            )
        return openai.OpenAI(api_key=api_key)
    if provider == "anthropic":
        import anthropic

        api_key = env.str("ANTHROPIC_API_KEY", None)
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Please copy env.example to tests/.env and add your API keys."
            )
        return anthropic.Anthropic(api_key=api_key)
    if provider == "gemini":
        import google.genai as genai

        api_key = env.str("GOOGLE_API_KEY", None)
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY environment variable not set. "
                "Please copy env.example to tests/.env and add your API keys."
            )
        return genai.Client(api_key=api_key)
    if provider == "bedrock":
        import boto3
        from botocore.exceptions import NoCredentialsError

        # Get AWS credentials from environment or .env file
        aws_access_key_id = env.str("AWS_ACCESS_KEY_ID", None)
        aws_secret_access_key = env.str("AWS_SECRET_ACCESS_KEY", None)
        region = env.str("AWS_DEFAULT_REGION", "us-east-2")

        try:
            # Create session with explicit credentials if available
            if aws_access_key_id and aws_secret_access_key:
                session = boto3.Session(
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=region,
                )
            else:
                # Fall back to default credential chain (AWS CLI, IAM roles, etc.)
                session = boto3.Session()

            # Test credentials by attempting to get caller identity
            session.client("sts").get_caller_identity()

            return session.client("bedrock-runtime", region_name=region)
        except NoCredentialsError as e:
            raise RuntimeError(
                "AWS credentials not configured. "
                "Please copy env.example to tests/.env and add your AWS credentials."
            ) from e
    if provider == "deepseek":
        import openai

        api_key = env.str("DEEPSEEK_API_KEY", None)
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY environment variable not set. "
                "Please copy env.example to tests/.env and add your API keys."
            )
        return openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    if provider == "grok":
        import openai

        api_key = env.str("XAI_API_KEY", None)
        if not api_key:
            raise RuntimeError(
                "XAI_API_KEY environment variable not set. "
                "Please copy env.example to tests/.env and add your API keys."
            )
        return openai.OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    if provider == "langchain":
        import openai

        api_key = env.str("OPENAI_API_KEY", None)
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable not set. "
                "Please copy env.example to tests/.env and add your API keys."
            )
        return openai.OpenAI(api_key=api_key)
    raise ValueError(f"Unsupported provider: {provider}")


def run_manual_test(provider: str, model: str, stream: bool = False) -> None:
    """Run manual test of a specific provider and model."""
    print(
        f"🧪 Manual testing: {provider} / {model} {'(streaming)' if stream else '(non-streaming)'}"
    )
    print("=" * 60)

    # Get API key for LLMCosts tracking (required)
    api_key = env.str("LLMCOSTS_API_KEY", None)
    if not api_key:
        raise RuntimeError(
            "LLMCOSTS_API_KEY environment variable not set. "
            "Please copy env.example to tests/.env and add your API key."
        )
    print(f"Using LLMCosts API key: {api_key[:8]}...")

    client = build_client(provider)

    # Map provider string to Provider enum
    provider_map = {
        "openai": Provider.OPENAI,
        "anthropic": Provider.ANTHROPIC,
        "gemini": Provider.GOOGLE,
        "bedrock": Provider.AMAZON_BEDROCK,
        "deepseek": Provider.DEEPSEEK,
        "grok": Provider.XAI,
        "langchain": Provider.OPENAI,  # LangChain uses OpenAI underneath
    }

    provider_enum = provider_map.get(provider)
    if not provider_enum:
        raise ValueError(f"Unknown provider: {provider}")

    proxy = LLMTrackingProxy(
        client, provider=provider_enum, debug=True, sync_mode=True, api_key=api_key
    )

    try:
        print(f"\n📞 Making API call to {provider}...")

        if provider == "gemini":
            if stream:
                print("Streaming Gemini generation...")
                stream_obj = proxy.models.generate_content_stream(
                    model=model, contents="Count from 1 to 3, one number per line."
                )
                chunks_count = 0
                for chunk in stream_obj:
                    chunks_count += 1
                print(f"✅ Received {chunks_count} streaming chunks")
            else:
                print("Non-streaming Gemini generation...")
                response = proxy.models.generate_content(
                    model=model, contents="Say hello in exactly 3 words"
                )
                print(f"✅ Response received: {response.text[:50]}...")

        elif provider == "anthropic":
            if stream:
                print("Streaming Anthropic messages...")
                stream_obj = proxy.messages.create(
                    model=model,
                    max_tokens=50,
                    messages=[
                        {
                            "role": "user",
                            "content": "Count from 1 to 3, one number per line.",
                        }
                    ],
                    stream=True,
                )
                chunks_count = 0
                for chunk in stream_obj:
                    chunks_count += 1
                print(f"✅ Received {chunks_count} streaming chunks")
            else:
                print("Non-streaming Anthropic messages...")
                response = proxy.messages.create(
                    model=model,
                    max_tokens=50,
                    messages=[
                        {"role": "user", "content": "Say hello in exactly 3 words"}
                    ],
                )
                print(f"✅ Response received: {response.content[0].text[:50]}...")

        elif provider == "bedrock":
            if stream:
                print("Streaming Bedrock converse...")
                stream_obj = proxy.converse_stream(
                    modelId=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"text": "Count from 1 to 3, one number per line."}
                            ],
                        }
                    ],
                    inferenceConfig={"maxTokens": 50},
                )
                chunks_count = 0
                for chunk in stream_obj:
                    chunks_count += 1
                print(f"✅ Received {chunks_count} streaming chunks")
            else:
                print("Non-streaming Bedrock converse...")
                response = proxy.converse(
                    modelId=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [{"text": "Say hello in exactly 3 words"}],
                        }
                    ],
                    inferenceConfig={"maxTokens": 20},
                )
                print(
                    f"✅ Response received: {response['output']['message']['content'][0]['text'][:50]}..."
                )

        elif provider == "langchain":
            try:
                from langchain_core.messages import HumanMessage
                from langchain_openai import ChatOpenAI
            except ImportError:
                raise RuntimeError(
                    "langchain-openai not installed. "
                    "Please install it with: pip install langchain-openai"
                )

            # Enable LangChain mode for compatibility
            proxy.enable_langchain_mode()

            # Create LangChain model using the tracked chat completions client
            langchain_model = ChatOpenAI(
                client=proxy.chat.completions,
                model=model,
                max_tokens=50,
                temperature=0.1,
                streaming=stream,
            )

            if stream:
                print("Streaming LangChain chat completion...")
                messages = [
                    HumanMessage(content="Count from 1 to 3, one number per line.")
                ]
                chunks_count = 0
                for chunk in langchain_model.stream(messages):
                    chunks_count += 1
                print(f"✅ Received {chunks_count} streaming chunks")
            else:
                print("Non-streaming LangChain chat completion...")
                messages = [HumanMessage(content="Say hello in exactly 3 words")]
                response = langchain_model.invoke(messages)
                print(f"✅ Response received: {response.content[:50]}...")

        else:  # openai, deepseek, grok
            if stream:
                print(f"Streaming {provider} chat completion...")
                stream_obj = proxy.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": "Count from 1 to 3, one number per line.",
                        }
                    ],
                    stream=True,
                    stream_options={"include_usage": True},
                    max_tokens=50,
                )
                chunks_count = 0
                for chunk in stream_obj:
                    chunks_count += 1
                print(f"✅ Received {chunks_count} streaming chunks")
            else:
                print(f"Non-streaming {provider} chat completion...")
                response = proxy.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": "Say hello in exactly 3 words"}
                    ],
                    max_tokens=20,
                )
                print(
                    f"✅ Response received: {response.choices[0].message.content[:50]}..."
                )

    except Exception as e:
        print(f"❌ API call failed: {e}")
        return

    finally:
        # Get the server response from the global tracker
        print("\n📊 Getting endpoint response...")
        try:
            from llmcosts.tracker.usage_delivery import get_usage_tracker

            tracker = get_usage_tracker()
            response = tracker.get_last_response()
            print("\n--- LLMCosts Endpoint Response ---")
            if response:
                print(json.dumps(response, indent=2))

                # Validate response structure
                if "status" in response and response["status"] == "success":
                    print("✅ Endpoint response successful")
                    if "events" in response and response["events"]:
                        print(f"✅ Cost events received: {len(response['events'])}")
                        cost_event = response["events"][0]
                        if "total_cost" in cost_event:
                            print(f"💰 Total cost: ${cost_event['total_cost']}")
                    else:
                        print("⚠️  No cost events in response")
                else:
                    print(f"⚠️  Response status: {response.get('status', 'unknown')}")
            else:
                print("❌ No server response (check API key validity)")
            tracker.shutdown()
        except Exception as e:
            print(f"❌ Error getting endpoint response: {e}")

    print("\n✅ Manual test completed!")


def run_automated_tests(args) -> int:
    """Run automated pytest tests with the specified options."""
    print("🤖 Running automated tests...")
    print("=" * 60)

    # Build pytest command
    cmd = ["uv", "run", "pytest"]

    # Add verbosity
    if args.verbose:
        cmd.append("-v")

    # Add capture settings
    if args.capture:
        cmd.append("-s")

    # Add collect-only flag
    if args.collect_only:
        cmd.append("--collect-only")

    # Build test file patterns
    if args.test_files:
        if (
            args.test_provider
            or args.streaming_only
            or args.nonstreaming_only
            or args.infrastructure
        ):
            print("Error: Cannot specify filtering flags with explicit test files")
            return 1
        cmd.extend(args.test_files)
    else:
        test_patterns = []

        if args.infrastructure:
            # Only run infrastructure tests
            test_patterns.extend(
                [
                    "tests/test_endpoint_integration.py",
                    "tests/test_endpoint_with_real_llms.py",
                    "tests/test_save_and_cost_events.py",
                    "tests/test_proxy_new_features.py",
                    "tests/test_usage_tracker.py",
                    "tests/test_logging.py",
                    "tests/test_thread_safety.py",
                ]
            )
        else:
            # Build provider test patterns
            providers = (
                [args.test_provider]
                if args.test_provider
                else [
                    "openai",
                    "anthropic",
                    "gemini",
                    "bedrock",
                    "deepseek",
                    "grok",
                    "langchain",
                ]
            )

            for provider in providers:
                if args.streaming_only:
                    test_patterns.append(f"tests/test_{provider}_streaming.py")
                elif args.nonstreaming_only:
                    test_patterns.append(f"tests/test_{provider}_nonstreaming.py")
                else:
                    # Both streaming and non-streaming
                    test_patterns.extend(
                        [
                            f"tests/test_{provider}_nonstreaming.py",
                            f"tests/test_{provider}_streaming.py",
                        ]
                    )

        # Add the test patterns to command
        cmd.extend(test_patterns)

    # Change to project root directory
    project_root = Path(__file__).parent.parent
    original_cwd = os.getcwd()
    os.chdir(project_root)

    print(f"Running command: {' '.join(cmd)}")
    print(f"Working directory: {os.getcwd()}")
    print("=" * 60)

    # Check for API keys
    env_file = Path("tests/.env")
    if not env_file.exists():
        print("Warning: tests/.env file not found.")
        print("Copy tests/env.example to tests/.env and add your API keys.")
        print("=" * 60)

    # Show what will be tested
    if not args.test_files:
        if args.infrastructure:
            print("🔧 Running infrastructure tests only")
        else:
            providers = (
                [args.test_provider]
                if args.test_provider
                else [
                    "openai",
                    "anthropic",
                    "gemini",
                    "bedrock",
                    "deepseek",
                    "grok",
                    "langchain",
                ]
            )
            mode_desc = (
                "streaming only"
                if args.streaming_only
                else "non-streaming only"
                if args.nonstreaming_only
                else "both streaming and non-streaming"
            )
            print(f"🎯 Testing providers: {', '.join(providers)} ({mode_desc})")
        print("=" * 60)

    # Run the tests
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTest run interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1
    finally:
        os.chdir(original_cwd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLMCosts testing and validation tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Test mode selection
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run automated pytest tests instead of manual testing",
    )

    # Manual testing arguments (only used when --test is not specified)
    parser.add_argument(
        "provider",
        nargs="?",
        choices=[
            "openai",
            "anthropic",
            "gemini",
            "bedrock",
            "deepseek",
            "grok",
            "langchain",
        ],
        help="Provider to test (required for manual testing)",
    )
    parser.add_argument(
        "model", nargs="?", help="Model ID to call (required for manual testing)"
    )
    parser.add_argument(
        "--stream", action="store_true", help="Use streaming mode for manual testing"
    )

    # Automated testing arguments (only used when --test is specified)
    parser.add_argument(
        "--infrastructure",
        action="store_true",
        help="Only run infrastructure tests (endpoint, proxy, logging, etc.)",
    )
    parser.add_argument(
        "--test-provider",
        choices=[
            "openai",
            "anthropic",
            "gemini",
            "bedrock",
            "deepseek",
            "grok",
            "langchain",
        ],
        help="Run tests for a specific LLM provider (for automated testing)",
    )
    parser.add_argument(
        "--streaming-only",
        action="store_true",
        help="Only run streaming tests",
    )
    parser.add_argument(
        "--nonstreaming-only",
        action="store_true",
        help="Only run non-streaming tests",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--capture",
        "-s",
        action="store_true",
        help="Don't capture output (useful for debugging)",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Only collect tests, don't run them",
    )
    parser.add_argument(
        "test_files",
        nargs="*",
        help="Specific test files to run (for automated testing)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.test:
        # Automated testing mode
        if args.streaming_only and args.nonstreaming_only:
            print("Error: Cannot specify both --streaming-only and --nonstreaming-only")
            sys.exit(1)

        # Arguments that don't make sense in test mode
        if args.stream:
            print("Error: --stream is only for manual testing, not --test mode")
            sys.exit(1)

        return run_automated_tests(args)
    else:
        # Manual testing mode
        if not args.provider or not args.model:
            print("Error: Provider and model are required for manual testing")
            print("Use --help to see usage examples")
            sys.exit(1)

        # Arguments that don't make sense in manual mode
        test_only_args = [
            args.infrastructure,
            args.streaming_only,
            args.nonstreaming_only,
            args.verbose,
            args.capture,
            args.collect_only,
            args.test_files,
        ]
        if any(test_only_args):
            print("Error: Test-specific arguments can only be used with --test flag")
            sys.exit(1)

        try:
            run_manual_test(args.provider, args.model, args.stream)
        except Exception as e:
            print(f"❌ Manual test failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
