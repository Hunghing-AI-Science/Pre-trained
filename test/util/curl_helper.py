"""
Utility to print curl commands for HTTP requests

This helps developers reproduce API calls using curl.
"""

import json
from typing import Dict, Any


def print_curl_command(url: str, headers: Dict[str, str], payload: Dict[str, Any], method: str = "POST"):
    """
    Print equivalent curl command for an HTTP request

    Args:
        url: Request URL
        headers: Request headers
        payload: Request body (JSON)
        method: HTTP method (default: POST)
    """
    payload_json = json.dumps(payload, indent=2)

    # Build curl command parts
    curl_parts = [f"curl -X {method} '{url}'"]

    # Add headers
    for key, value in headers.items():
        curl_parts.append(f"  -H '{key}: {value}'")

    # Add payload (escape single quotes for shell)
    if payload:
        payload_json_escaped = payload_json.replace("'", "'\\''")
        curl_parts.append(f"  -d '{payload_json_escaped}'")

    curl_command = " \\\n".join(curl_parts)

    print(f"\n🔧 Equivalent curl command:")
    print("=" * 80)
    print(curl_command)
    print("=" * 80)


def print_curl_for_chat_completion(
    url: str = "http://localhost:8000/v1/chat/completions",
    api_key: str = "test-key",
    model: str = "gpt-3.5-turbo",
    messages: list = None,
    temperature: float = 0.7,
    max_tokens: int = 150
):
    """
    Print curl command for chat completion request

    Args:
        url: API endpoint URL
        api_key: API authorization key
        model: Model to use
        messages: List of chat messages
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
    """
    if messages is None:
        messages = [{"role": "user", "content": "Hello!"}]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    print_curl_command(url, headers, payload)


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("Example: Chat Completion curl command")
    print("=" * 80)

    print_curl_for_chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Python?"}
        ]
    )

    print("\n" + "=" * 80)
    print("You can copy and run the above curl command in your terminal!")
    print("=" * 80)

