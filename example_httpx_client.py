"""
Simple example of using httpx AsyncClient for GPT async API polling

This demonstrates the recommended way to use the async GPT API with httpx.

Requirements:
    pip install httpx asyncio

Usage:
    python example_httpx_client.py
"""

import httpx
import asyncio


async def create_task(base_url: str, api_key: str, messages: list, model: str = "gpt-3.5-turbo"):
    """Create a chat completion task and return task ID"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 150
            },
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()["id"]


async def get_task_status(base_url: str, api_key: str, task_id: str):
    """Get the status of a task"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{base_url}/v1/chat/tasks/{task_id}",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()


async def poll_until_complete(base_url: str, api_key: str, task_id: str, timeout: int = 300):
    """Poll task until completion"""
    import time
    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise TimeoutError(f"Task did not complete within {timeout}s")

        status_data = await get_task_status(base_url, api_key, task_id)
        status = status_data["status"]

        if status == "completed":
            return status_data["result"]
        elif status == "failed":
            raise Exception(f"Task failed: {status_data.get('error')}")

        # Wait before next poll
        await asyncio.sleep(2)


async def delete_task(base_url: str, api_key: str, task_id: str):
    """Delete a completed task"""
    async with httpx.AsyncClient() as client:
        response = await client.delete(
            f"{base_url}/v1/chat/tasks/{task_id}",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()


async def chat_completion(
    base_url: str,
    api_key: str,
    messages: list,
    model: str = "gpt-3.5-turbo",
    cleanup: bool = True
):
    """
    High-level async chat completion with httpx

    Args:
        base_url: API base URL
        api_key: API key for authentication
        messages: List of message dicts with 'role' and 'content'
        model: Model to use
        cleanup: Whether to delete task after getting result

    Returns:
        Completion result
    """
    # Create task
    print(f"Creating task...")
    task_id = await create_task(base_url, api_key, messages, model)
    print(f"Task created: {task_id}")

    # Poll for result
    print(f"Polling for result...")
    result = await poll_until_complete(base_url, api_key, task_id)
    print(f"Task completed!")

    # Cleanup
    if cleanup:
        await delete_task(base_url, api_key, task_id)
        print(f"Task deleted")

    return result


async def main():
    """Example usage"""
    # Configuration
    BASE_URL = "http://localhost:8000"
    API_KEY = "test-key"

    # Simple question
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python? Answer in one sentence."}
    ]

    print("=" * 80)
    print("GPT Async API Example with httpx")
    print("=" * 80)

    try:
        # Get completion
        result = await chat_completion(BASE_URL, API_KEY, messages)

        # Display result
        print("\n" + "=" * 80)
        print("RESULT")
        print("=" * 80)
        print(f"Response: {result['choices'][0]['message']['content']}")
        print(f"\nTokens used:")
        print(f"  - Prompt: {result['usage']['prompt_tokens']}")
        print(f"  - Completion: {result['usage']['completion_tokens']}")
        print(f"  - Total: {result['usage']['total_tokens']}")
        print("=" * 80)

    except httpx.HTTPStatusError as e:
        print(f"HTTP Error: {e.response.status_code}")
        print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"Error: {e}")


async def example_concurrent():
    """Example: Process multiple completions concurrently"""
    BASE_URL = "http://localhost:8000"
    API_KEY = "test-key"

    questions = [
        "What is 2+2?",
        "What is the capital of France?",
        "What programming language is named after a snake?"
    ]

    print("\n" + "=" * 80)
    print("Concurrent Completions Example")
    print("=" * 80)

    # Create all tasks concurrently
    print("Creating tasks concurrently...")
    task_ids = await asyncio.gather(*[
        create_task(
            BASE_URL,
            API_KEY,
            [{"role": "user", "content": q}],
            "gpt-3.5-turbo"
        )
        for q in questions
    ])
    print(f"Created {len(task_ids)} tasks")

    # Poll all tasks concurrently
    print("Polling tasks concurrently...")
    results = await asyncio.gather(*[
        poll_until_complete(BASE_URL, API_KEY, task_id)
        for task_id in task_ids
    ])

    # Display results
    print("\nResults:")
    for i, (question, result) in enumerate(zip(questions, results), 1):
        answer = result['choices'][0]['message']['content']
        print(f"\n{i}. Q: {question}")
        print(f"   A: {answer}")

    # Cleanup
    print("\nCleaning up...")
    await asyncio.gather(*[
        delete_task(BASE_URL, API_KEY, task_id)
        for task_id in task_ids
    ])
    print("All tasks deleted")


if __name__ == "__main__":
    # Run simple example
    asyncio.run(main())

    # Uncomment to run concurrent example
    # asyncio.run(example_concurrent())

