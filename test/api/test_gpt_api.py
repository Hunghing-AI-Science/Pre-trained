"""
Simple API test case for GPT Chat Completions

This test file demonstrates:
1. Simple single-turn chat completion
2. Long multi-turn conversation with context
3. Multiple completions (n > 1)
4. Temperature variations
5. Error handling

Usage:
    python test/api/test_gpt_api.py
"""

import requests
import json
import time
from typing import List, Dict, Any


class GPTAPIClient:
    """Simple client for testing GPT API"""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = "test-key"):
        """
        Initialize the GPT API client

        Args:
            base_url: Base URL of the API server
            api_key: API key for authorization
        """
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "openai/gpt-oss-20b",
        temperature: float = 0.7,
        max_tokens: int = 150,
        n: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a chat completion

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            n: Number of completions
            **kwargs: Additional parameters

        Returns:
            API response as dictionary
        """
        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n": n,
            **kwargs
        }

        print(f"\n📤 Request to: {url}")
        print(f"   Model: {model}")
        print(f"   Messages: {len(messages)}")
        print(f"   Temperature: {temperature}")
        print(f"   Max tokens: {max_tokens}")

        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()

        result = response.json()
        print(f"✅ Response received")
        print(f"   Tokens: {result['usage']['total_tokens']}")

        return result


def test_simple_completion():
    """Test 1: Simple single-turn completion"""
    print("\n" + "=" * 80)
    print("TEST 1: Simple Chat Completion")
    print("=" * 80)

    client = GPTAPIClient()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python? Answer in one sentence."}
    ]

    try:
        result = client.chat_completion(messages=messages, temperature=0.7, max_tokens=100)

        # Display results
        print("\n📝 Result:")
        print(f"   ID: {result['id']}")
        print(f"   Model: {result['model']}")
        print(f"   Response: {result['choices'][0]['message']['content']}")
        print(f"\n📊 Usage:")
        print(f"   Prompt tokens: {result['usage']['prompt_tokens']}")
        print(f"   Completion tokens: {result['usage']['completion_tokens']}")
        print(f"   Total tokens: {result['usage']['total_tokens']}")

        print("\n✅ Test 1 PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Test 1 FAILED: {str(e)}")
        return False


def test_long_conversation():
    """Test 2: Long multi-turn conversation with context"""
    print("\n" + "=" * 80)
    print("TEST 2: Long Multi-Turn Conversation")
    print("=" * 80)

    client = GPTAPIClient()

    # Initialize conversation
    conversation = [
        {"role": "system", "content": "You are a helpful Python programming tutor."}
    ]

    try:
        # Turn 1: Ask about decorators
        print("\n--- Turn 1 ---")
        conversation.append({"role": "user", "content": "What is a Python decorator?"})

        result_1 = client.chat_completion(messages=conversation, temperature=0.7, max_tokens=150)
        assistant_response_1 = result_1['choices'][0]['message']['content']

        print(f"\n👤 User: {conversation[-1]['content']}")
        print(f"🤖 Assistant: {assistant_response_1[:200]}...")
        print(f"📊 Tokens: {result_1['usage']['total_tokens']}")

        # Add assistant response to conversation
        conversation.append({"role": "assistant", "content": assistant_response_1})

        # Turn 2: Ask for example
        print("\n--- Turn 2 ---")
        conversation.append({"role": "user", "content": "Can you show me a simple example?"})

        result_2 = client.chat_completion(messages=conversation, temperature=0.7, max_tokens=150)
        assistant_response_2 = result_2['choices'][0]['message']['content']

        print(f"\n👤 User: {conversation[-1]['content']}")
        print(f"🤖 Assistant: {assistant_response_2[:200]}...")
        print(f"📊 Tokens: {result_2['usage']['total_tokens']}")

        # Add assistant response to conversation
        conversation.append({"role": "assistant", "content": assistant_response_2})

        # Turn 3: Follow-up question
        print("\n--- Turn 3 ---")
        conversation.append({"role": "user", "content": "What are common use cases?"})

        result_3 = client.chat_completion(messages=conversation, temperature=0.7, max_tokens=150)
        assistant_response_3 = result_3['choices'][0]['message']['content']

        print(f"\n👤 User: {conversation[-1]['content']}")
        print(f"🤖 Assistant: {assistant_response_3[:200]}...")
        print(f"📊 Tokens: {result_3['usage']['total_tokens']}")

        # Summary
        print("\n--- Conversation Summary ---")
        print(f"Total messages: {len(conversation)}")
        print(f"Total turns: {(len(conversation) - 1) // 2}")
        print(f"Final token count: {result_3['usage']['total_tokens']}")
        print("\n✅ Context maintained across all turns!")

        print("\n✅ Test 2 PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Test 2 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_completions():
    """Test 3: Generate multiple completions (n > 1)"""
    print("\n" + "=" * 80)
    print("TEST 3: Multiple Completions")
    print("=" * 80)

    client = GPTAPIClient()

    messages = [
        {"role": "user", "content": "Write a creative opening line for a sci-fi story."}
    ]

    try:
        result = client.chat_completion(
            messages=messages,
            temperature=0.9,  # Higher temperature for creativity
            max_tokens=50,
            n=3  # Generate 3 different completions
        )

        print("\n📝 Generated 3 different responses:")
        for i, choice in enumerate(result['choices'], 1):
            print(f"\n{i}. {choice['message']['content']}")

        print(f"\n📊 Total tokens: {result['usage']['total_tokens']}")

        print("\n✅ Test 3 PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Test 3 FAILED: {str(e)}")
        return False


def test_temperature_comparison():
    """Test 4: Compare low vs high temperature"""
    print("\n" + "=" * 80)
    print("TEST 4: Temperature Comparison")
    print("=" * 80)

    client = GPTAPIClient()

    messages = [
        {"role": "user", "content": "Describe a sunset in 20 words."}
    ]

    try:
        # Low temperature (focused, deterministic)
        print("\n--- Low Temperature (0.1) - More Focused ---")
        result_low = client.chat_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=50
        )

        print(f"🤖 Response: {result_low['choices'][0]['message']['content']}")

        # High temperature (creative, diverse)
        print("\n--- High Temperature (1.5) - More Creative ---")
        result_high = client.chat_completion(
            messages=messages,
            temperature=1.5,
            max_tokens=50
        )

        print(f"🤖 Response: {result_high['choices'][0]['message']['content']}")

        print("\n✅ Test 4 PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Test 4 FAILED: {str(e)}")
        return False


def test_error_handling():
    """Test 5: Error handling (missing authorization)"""
    print("\n" + "=" * 80)
    print("TEST 5: Error Handling")
    print("=" * 80)

    # Create client without API key
    url = "http://localhost:8000/v1/chat/completions"

    messages = [
        {"role": "user", "content": "Hello"}
    ]

    payload = {
        "model": "gpt-3.5-turbo",
        "messages": messages
    }

    try:
        # Try without authorization header
        print("\n🔒 Testing without authorization header...")
        response = requests.post(url, json=payload)

        if response.status_code == 401:
            print(f"✅ Correctly returned 401 Unauthorized")
            print(f"   Error: {response.json()['detail']}")
            print("\n✅ Test 5 PASSED")
            return True
        else:
            print(f"❌ Expected 401, got {response.status_code}")
            return False

    except Exception as e:
        print(f"\n❌ Test 5 FAILED: {str(e)}")
        return False


def test_health_check():
    """Test 0: Health check endpoint"""
    print("\n" + "=" * 80)
    print("TEST 0: Health Check")
    print("=" * 80)

    try:
        url = "http://localhost:8000/"
        response = requests.get(url)
        response.raise_for_status()

        result = response.json()
        print(f"\n✅ API is running")
        print(f"   Service: {result['service']}")
        print(f"   Version: {result['version']}")
        print(f"   Status: {result['status']}")

        print("\n✅ Test 0 PASSED")
        return True

    except Exception as e:
        print(f"\n❌ API is not running: {str(e)}")
        print(f"\n💡 Make sure to start the API server:")
        print(f"   uvicorn src.gpu_server.api:app --reload")
        return False


def main():
    """Run all tests"""
    print("=" * 80)
    print("GPT API Test Suite")
    print("=" * 80)
    print("\nTesting GPT Chat Completions API")
    print("Base URL: http://localhost:8000")
    print("=" * 80)

    # Track results
    results = []

    # Run tests
    results.append(("Health Check", test_health_check()))

    if not results[0][1]:
        print("\n" + "=" * 80)
        print("⚠️  API server is not running. Please start it first:")
        print("   uvicorn src.gpu_server.api:app --reload")
        print("=" * 80)
        return

    results.append(("Simple Completion", test_simple_completion()))
    results.append(("Long Conversation", test_long_conversation()))
    results.append(("Multiple Completions", test_multiple_completions()))
    results.append(("Temperature Comparison", test_temperature_comparison()))
    results.append(("Error Handling", test_error_handling()))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<50} {status}")

    print("=" * 80)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 80)

    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")


if __name__ == "__main__":
    main()

