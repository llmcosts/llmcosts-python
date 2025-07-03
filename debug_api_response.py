#!/usr/bin/env python3

import json
import os

import requests
from environs import Env

# Load environment variables
env = Env()
env.read_env(os.path.join(os.path.dirname(__file__), "tests", ".env"))


def debug_api_response():
    """Debug the actual API response structure."""
    api_key = env.str("LLMCOSTS_API_KEY", None)
    if not api_key:
        print("❌ LLMCOSTS_API_KEY not found")
        return

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Test payload similar to the failing test
    payload = {
        "usage_records": [
            {
                "model_id": "gpt-4o-mini",
                "provider": "openai",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                "response_id": "debug-test-12345",
                "client_customer_key": None,
                "timestamp": "2025-01-01T00:00:00Z",
            }
        ],
        "remote_save": False,  # Don't save to avoid cluttering database
    }

    print("📤 Sending request to /api/v1/usage:")
    print(json.dumps(payload, indent=2))

    # Make the request
    res = requests.post(
        "https://llmcosts.com/api/v1/usage", json=payload, headers=headers
    )

    print(f"\n📥 Response status: {res.status_code}")

    if res.status_code == 200:
        response_data = res.json()
        print("\n📄 Full response:")
        print(json.dumps(response_data, indent=2))

        if "events" in response_data and response_data["events"]:
            event = response_data["events"][0]
            print(f"\n🔍 Event fields: {list(event.keys())}")
            print(f"📋 Has client_customer_key: {'client_customer_key' in event}")
            if "client_customer_key" in event:
                print(f"💡 client_customer_key value: {event['client_customer_key']}")

            # Test without client_customer_key field at all
            print("\n" + "=" * 50)
            print("Testing WITHOUT client_customer_key field:")
            payload2 = {
                "usage_records": [
                    {
                        "model_id": "gpt-4o-mini",
                        "provider": "openai",
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        },
                        "response_id": "debug-test-no-field-67890",
                        "timestamp": "2025-01-01T00:00:00Z",
                        # No client_customer_key field at all
                    }
                ],
                "remote_save": False,
            }

            res2 = requests.post(
                "https://llmcosts.com/api/v1/usage", json=payload2, headers=headers
            )

            if res2.status_code == 200:
                response_data2 = res2.json()
                if "events" in response_data2 and response_data2["events"]:
                    event2 = response_data2["events"][0]
                    print(f"🔍 Event fields (no field sent): {list(event2.keys())}")
                    print(
                        f"📋 Has client_customer_key: {'client_customer_key' in event2}"
                    )
                    if "client_customer_key" in event2:
                        print(
                            f"💡 client_customer_key value: {event2['client_customer_key']}"
                        )
        else:
            print("❌ No events in response")
    else:
        print(f"❌ Request failed: {res.text}")


if __name__ == "__main__":
    debug_api_response()
