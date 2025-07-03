"""
A sample project to demonstrate the usage of LLMTrackingProxy and cost limits.

This script is intended to be run manually.
"""

import logging
import os
import sys
from pathlib import Path
from time import sleep

import openai
from environs import Env

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llmcosts import thresholds
from llmcosts.exceptions import TriggeredLimitError
from llmcosts.tracker import LLMTrackingProxy
from llmcosts.tracker.providers import Provider

# Configure logging to see debug information
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Load environment variables from the .env file in the tests directory
env = Env()
# Construct the path to the .env file located in the parent 'tests' directory
env_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
)
env.read_env(env_path)


def print_decrypted_triggered_thresholds(proxy_instance, call_description):
    """Print decrypted triggered thresholds for debugging purposes."""
    print(f"\n=== Triggered Thresholds Debug - {call_description} ===")
    try:
        # Check if client has triggered thresholds
        has_thresholds = proxy_instance._llm_costs_client.has_triggered_thresholds
        print(f"  Has triggered thresholds: {has_thresholds}")

        if has_thresholds:
            version = proxy_instance._llm_costs_client.triggered_thresholds_version
            print(f"  Version: {version}")

            # Get and decrypt triggered thresholds
            decrypted = (
                proxy_instance._llm_costs_client.get_decrypted_triggered_thresholds()
            )
            if decrypted:
                print(f"  Decrypted payload: {decrypted}")
                triggered_list = decrypted.get("triggered_thresholds", [])
                print(f"  Number of triggered thresholds: {len(triggered_list)}")
                for i, threshold in enumerate(triggered_list):
                    print(
                        f"    Threshold {i + 1}: {threshold.get('threshold_type', 'unknown')} - ${threshold.get('amount', 'unknown')} - {threshold.get('period', 'unknown')}"
                    )
            else:
                print("  Could not decrypt triggered thresholds")
        else:
            print("  No triggered thresholds found")
    except Exception as e:
        print(f"  Error checking triggered thresholds: {e}")
    print("=" * 60)


# --- 1. Configuration ---
LLMCOSTS_API_KEY = env.str("LLMCOSTS_API_KEY", None)
OPENAI_API_KEY = env.str("OPENAI_API_KEY", None)

if not LLMCOSTS_API_KEY or not OPENAI_API_KEY:
    print("Error: LLMCOSTS_API_KEY and OPENAI_API_KEY must be set in tests/.env")
    sys.exit(1)

# --- 2. Instantiate LLMTrackingProxy ---
print("Instantiating LLMTrackingProxy...")
openai_target_client = openai.OpenAI(api_key=OPENAI_API_KEY)
my_proxy = LLMTrackingProxy(
    target=openai_target_client,
    provider=Provider.OPENAI,
    client_customer_key="1234",
    api_key=LLMCOSTS_API_KEY,
)

# Print initial triggered thresholds state
print_decrypted_triggered_thresholds(my_proxy, "Initial State")

# --- 3. Make initial calls ---
print("\n--- Making three initial API calls ---")
for i in range(3):
    print(f"Call {i + 1}...")
    try:
        response = my_proxy.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "Hi"}]
        )
        print(
            f"  Response {i + 1} received. Choice: {response.choices[0].message.content}"
        )
        # Print triggered thresholds after each call
        print_decrypted_triggered_thresholds(my_proxy, f"After API Call {i + 1}")
    except Exception as e:
        print(f"  An error occurred: {e}")


# --- 4. Create a limit threshold ---
print("\n--- Creating a limit threshold for gpt-4o-mini ---")
# try:
# limit_payload = {
#     "threshold_type": "limit",
#     "amount": "0.05",
#     "period": "daily",
#     "provider": "openai",
#     "model_id": "gpt-4o-mini",
#     "client_customer_key": "1234",
# }
limit_payload = {
    "threshold_type": "limit",
    "amount": "0.0020",
    "period": "day",  # Changed from "daily" to "day"
    "provider": "openai",
    "model_id": "gpt-4o-mini",  # Use a model that exists in the system
    # "client_customer_key": "valid_customer_key",  # Use an existing customer key
}
created_threshold = thresholds.create_threshold(
    my_proxy._llm_costs_client, limit_payload
)
#     print("  Successfully created limit threshold:")
#     print(f"    ID: {created_threshold.get('id')}")
#     print(f"    Amount: ${created_threshold.get('amount')}")
# except HTTPStatusError as e:
#     print(f"  An HTTP error occurred while creating threshold: {e}")
#     print(f"  Response body: {e.response.text}")
# except Exception as e:
#     print(f"  An unexpected error occurred while creating threshold: {e}")

print_decrypted_triggered_thresholds(my_proxy, "After Creating Threshold")

sleep(10)
# --- 5. Make another call (should now trigger limit) ---
print("\n--- Making another call (should trigger the limit) ---")
try:
    response = my_proxy.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "Hi again"}]
    )
    print(f"  Response received. Choice: {response.choices[0].message.content}")
    print_decrypted_triggered_thresholds(my_proxy, "After Triggering Limit Call")
except TriggeredLimitError as e:
    print(f"  Caught an expected TriggeredLimitError: {e}")
    print_decrypted_triggered_thresholds(my_proxy, "After TriggeredLimitError")
except Exception as e:
    print(f"  An unexpected error occurred: {e}")


# --- 6. Make a final call (should see an exception) ---
print("\n--- Making a final call (should see an exception) ---")
try:
    response = my_proxy.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "Hi one last time"}]
    )
    print(f"  Response received. Choice: {response.choices[0].message.content}")
    print_decrypted_triggered_thresholds(my_proxy, "After Final Call")
except TriggeredLimitError as e:
    print(f"  Caught an expected TriggeredLimitError: {e}")
    print_decrypted_triggered_thresholds(my_proxy, "After Final TriggeredLimitError")
except Exception as e:
    print(f"  An unexpected error occurred: {e}")

print("\nScript finished.")
