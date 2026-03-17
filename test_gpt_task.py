"""
Test file demonstrating GPT task usage including long conversations
"""
import uuid
from datetime import datetime, timezone
from src.gpu_server.database import SessionLocal, GPTTask, init_db
from src.gpu_server.celery_app.tasks import process_gpt_task

def test_simple_gpt_completion():
    """Test a simple GPT completion"""
    print("\n" + "=" * 80)
    print("Test 1: Simple GPT Completion")
    print("=" * 80)

    db = SessionLocal()
    try:
        # Initialize database
        init_db()

        # Create a task in the database
        task_id = str(uuid.uuid4())
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Python?"}
        ]

        task = GPTTask(
            id=task_id,
            status="pending",
            messages=messages,
            model_name="openai/gpt-oss-20b",
            created_at=datetime.now(timezone.utc)
        )
        db.add(task)
        db.commit()

        print(f"✓ Created task: {task_id}")
        print(f"  Messages: {len(messages)}")

        # Process the task
        result = process_gpt_task(
            task_id=task_id,
            messages=messages,
            temperature=0.7,
            max_tokens=150
        )

        print(f"✓ Task completed")
        print(f"  Response: {result['result']['text'][:200]}...")
        print(f"  Tokens: {result['result']['usage']['total_tokens']}")

    finally:
        db.close()


def test_long_conversation():
    """Test a long multi-turn conversation"""
    print("\n" + "=" * 80)
    print("Test 2: Long Multi-Turn Conversation")
    print("=" * 80)

    db = SessionLocal()
    try:
        # Initialize the conversation
        conversation = [
            {"role": "system", "content": "You are a knowledgeable AI assistant helping with Python programming questions."}
        ]

        # Turn 1: Ask about list comprehensions
        print("\n--- Turn 1: User asks about list comprehensions ---")
        task_id_1 = str(uuid.uuid4())
        conversation.append({"role": "user", "content": "What is a list comprehension in Python?"})

        task_1 = GPTTask(
            id=task_id_1,
            status="pending",
            messages=conversation.copy(),
            model_name="openai/gpt-oss-20b",
            created_at=datetime.now(timezone.utc)
        )
        db.add(task_1)
        db.commit()

        result_1 = process_gpt_task(
            task_id=task_id_1,
            messages=conversation.copy(),
            temperature=0.7,
            max_tokens=150
        )

        print(f"User: {conversation[-1]['content']}")
        print(f"Assistant: {result_1['result']['text'][:200]}...")
        print(f"Tokens: {result_1['result']['usage']['total_tokens']}")

        # Add assistant's response to conversation
        conversation.append({"role": "assistant", "content": result_1['result']['text']})

        # Turn 2: Ask for an example
        print("\n--- Turn 2: User asks for an example ---")
        task_id_2 = str(uuid.uuid4())
        conversation.append({"role": "user", "content": "Can you show me a practical example?"})

        task_2 = GPTTask(
            id=task_id_2,
            status="pending",
            messages=conversation.copy(),
            model_name="openai/gpt-oss-20b",
            created_at=datetime.now(timezone.utc)
        )
        db.add(task_2)
        db.commit()

        result_2 = process_gpt_task(
            task_id=task_id_2,
            messages=conversation.copy(),
            temperature=0.7,
            max_tokens=150
        )

        print(f"User: {conversation[-1]['content']}")
        print(f"Assistant: {result_2['result']['text'][:200]}...")
        print(f"Tokens: {result_2['result']['usage']['total_tokens']}")

        # Add assistant's response to conversation
        conversation.append({"role": "assistant", "content": result_2['result']['text']})

        # Turn 3: Ask about performance
        print("\n--- Turn 3: User asks about performance ---")
        task_id_3 = str(uuid.uuid4())
        conversation.append({"role": "user", "content": "Is it faster than using a regular for loop?"})

        task_3 = GPTTask(
            id=task_id_3,
            status="pending",
            messages=conversation.copy(),
            model_name="openai/gpt-oss-20b",
            created_at=datetime.now(timezone.utc)
        )
        db.add(task_3)
        db.commit()

        result_3 = process_gpt_task(
            task_id=task_id_3,
            messages=conversation.copy(),
            temperature=0.7,
            max_tokens=150
        )

        print(f"User: {conversation[-1]['content']}")
        print(f"Assistant: {result_3['result']['text'][:200]}...")
        print(f"Tokens: {result_3['result']['usage']['total_tokens']}")

        # Display conversation summary
        print("\n--- Conversation Summary ---")
        print(f"Total turns: {(len(conversation) - 1) // 2}")  # Exclude system message
        print(f"Total messages in conversation: {len(conversation)}")
        print(f"Final context length: ~{result_3['result']['usage']['total_tokens']} tokens")
        print("\n✓ Long conversation test completed successfully!")

    finally:
        db.close()


def test_multiple_models():
    """Test using different GPT models"""
    print("\n" + "=" * 80)
    print("Test 3: Multiple Models")
    print("=" * 80)

    db = SessionLocal()
    try:
        messages = [
            {"role": "user", "content": "Write a haiku about coding."}
        ]

        # Test with 20B model
        print("\n--- 20B Model ---")
        task_id_20b = str(uuid.uuid4())
        task_20b = GPTTask(
            id=task_id_20b,
            status="pending",
            messages=messages,
            model_name="openai/gpt-oss-20b",
            created_at=datetime.now(timezone.utc)
        )
        db.add(task_20b)
        db.commit()

        result_20b = process_gpt_task(
            task_id=task_id_20b,
            messages=messages,
            model_name="openai/gpt-oss-20b",
            temperature=0.9,
            max_tokens=100
        )

        print(f"Model: {result_20b['result']['model']}")
        print(f"Response: {result_20b['result']['text']}")
        print(f"Tokens: {result_20b['result']['usage']['total_tokens']}")

        print("\n✓ Model switching test completed!")

    finally:
        db.close()


if __name__ == "__main__":
    print("=" * 80)
    print("GPT Task Tests")
    print("=" * 80)

    try:
        # Run all tests
        test_simple_gpt_completion()
        test_long_conversation()
        test_multiple_models()

        print("\n" + "=" * 80)
        print("All tests completed!")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Error running tests: {str(e)}")
        import traceback
        traceback.print_exc()

