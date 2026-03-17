"""
Test for GPT Chat Completions with Celery + Internal Polling

The API now:
1. Creates a Celery task for background processing
2. Internally polls the task status using httpx.AsyncClient() with asyncio.sleep()
3. Returns the completed result directly (no UUID exposed to client)

This provides the benefits of async processing while maintaining a simple client interface.

Usage:
    python test_gpt_sync.py
"""

import requests
import json


def print_curl_command(url, headers, payload):
    """Print equivalent curl command"""
    payload_json = json.dumps(payload, indent=2)

    # Build curl command
    curl_parts = [f"curl -X POST '{url}'"]

    for key, value in headers.items():
        curl_parts.append(f"  -H '{key}: {value}'")

    # Escape payload for shell
    payload_json_escaped = payload_json.replace("'", "'\\''")
    curl_parts.append(f"  -d '{payload_json_escaped}'")

    curl_command = " \\\n".join(curl_parts)

    print(f"\n🔧 Equivalent curl command:")
    print("=" * 80)
    print(curl_command)
    print("=" * 80)


def test_simple_completion():
    """Test simple chat completion with direct response"""
    print("=" * 80)
    print("GPT Synchronous API Test")
    print("=" * 80)

    url = "http://localhost:8000/v1/chat/completions"
    headers = {
        "Authorization": "Bearer test-key",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Python? Answer in one sentence."}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }

    print("\n📤 Sending request...")
    print(f"   Model: {payload['model']}")
    print(f"   Messages: {len(payload['messages'])}")

    # Print curl command
    print_curl_command(url, headers, payload)

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()

        result = response.json()

        print("\n✅ Response received directly!")
        print(f"\n📝 Result:")
        print(f"   ID: {result['id']}")
        print(f"   Model: {result['model']}")
        print(f"   Response: {result['choices'][0]['message']['content']}")
        print(f"\n📊 Usage:")
        print(f"   Prompt tokens: {result['usage']['prompt_tokens']}")
        print(f"   Completion tokens: {result['usage']['completion_tokens']}")
        print(f"   Total tokens: {result['usage']['total_tokens']}")

        print("\n✅ TEST PASSED")

    except requests.exceptions.Timeout:
        print("\n❌ Request timed out")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text}")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def test_conversation():
    """Test multi-turn conversation"""
    print("\n" + "=" * 80)
    print("Multi-Turn Conversation Test")
    print("=" * 80)

    url = "http://localhost:8000/v1/chat/completions"
    headers = {
        "Authorization": "Bearer test-key",
        "Content-Type": "application/json"
    }

    conversation = [
        {"role": "system", "content": "You are a helpful Python tutor."}
    ]

    try:
        # Turn 1
        print("\n--- Turn 1 ---")
        conversation.append({"role": "user", "content": "What is a list?"})

        payload_1 = {
            "model": "gpt-3.5-turbo",
            "messages": conversation,
            "temperature": 0.7,
            "max_tokens": 100
        }

        print_curl_command(url, headers, payload_1)

        response = requests.post(url, headers=headers, json=payload_1, timeout=120)
        response.raise_for_status()
        result = response.json()

        assistant_msg = result['choices'][0]['message']['content']
        print(f"\n👤 User: {conversation[-1]['content']}")
        print(f"🤖 Assistant: {assistant_msg[:150]}...")

        conversation.append({"role": "assistant", "content": assistant_msg})

        # Turn 2
        print("\n--- Turn 2 ---")
        conversation.append({"role": "user", "content": "Give me an example."})

        payload_2 = {
            "model": "gpt-3.5-turbo",
            "messages": conversation,
            "temperature": 0.7,
            "max_tokens": 100
        }

        print_curl_command(url, headers, payload_2)

        response = requests.post(url, headers=headers, json=payload_2, timeout=120)
        response.raise_for_status()
        result = response.json()

        assistant_msg = result['choices'][0]['message']['content']
        print(f"👤 User: {conversation[-1]['content']}")
        print(f"🤖 Assistant: {assistant_msg[:150]}...")

        print(f"\n✅ Conversation completed with {len(conversation)} messages")
        print("✅ TEST PASSED")

    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    print("\nMake sure the API server is running on http://localhost:8000\n")

    # Run tests
    test_simple_completion()
    test_conversation()

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)

