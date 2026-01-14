#!/usr/bin/env python3
"""
Test script to verify Celery worker can find and register GPT tasks
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_import_celery_app():
    """Test 1: Can we import the celery app?"""
    print("=" * 60)
    print("TEST 1: Importing Celery App")
    print("=" * 60)
    try:
        from src.gpu_server.celery_app.celery_app import celery_app
        print("✅ Successfully imported celery_app")
        print(f"   App name: {celery_app.main}")
        print(f"   Broker: {celery_app.conf.broker_url}")
        return celery_app
    except Exception as e:
        print(f"❌ Failed to import celery_app: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_task_registration(celery_app):
    """Test 2: Are tasks registered?"""
    print("\n" + "=" * 60)
    print("TEST 2: Task Registration")
    print("=" * 60)

    if not celery_app:
        print("❌ Skipping - celery_app not available")
        return False

    # Force task discovery
    print("Forcing task discovery...")
    celery_app.loader.import_default_modules()

    # List all registered tasks
    registered_tasks = list(celery_app.tasks.keys())
    print(f"\nFound {len(registered_tasks)} registered tasks:")

    for task_name in sorted(registered_tasks):
        if not task_name.startswith('celery.'):
            print(f"   📋 {task_name}")

    # Check for our specific task
    target_task = "src.gpu_server.celery_app.gpt_tasks.process_chat_completion"
    if target_task in registered_tasks:
        print(f"\n✅ Target task '{target_task}' is registered!")
        return True
    else:
        print(f"\n❌ Target task '{target_task}' NOT found!")
        print(f"   Available GPT tasks:")
        gpt_tasks = [t for t in registered_tasks if 'gpt' in t.lower()]
        if gpt_tasks:
            for t in gpt_tasks:
                print(f"      - {t}")
        else:
            print("      (none)")
        return False


def test_task_routing(celery_app):
    """Test 3: Is task routing configured?"""
    print("\n" + "=" * 60)
    print("TEST 3: Task Routing Configuration")
    print("=" * 60)

    if not celery_app:
        print("❌ Skipping - celery_app not available")
        return False

    # Check task routes
    task_routes = celery_app.conf.task_routes
    print(f"Task routes configured: {task_routes}")

    # Check task queues
    task_queues = celery_app.conf.task_queues
    if task_queues:
        print(f"\nConfigured queues:")
        for queue in task_queues:
            print(f"   📬 {queue.name}")

    target_task = "src.gpu_server.celery_app.gpt_tasks.process_chat_completion"
    if target_task in task_routes:
        print(f"\n✅ Task routing configured for '{target_task}'")
        print(f"   Route: {task_routes[target_task]}")
        return True
    else:
        print(f"\n⚠️  No explicit route for '{target_task}'")
        return True  # This might be okay if using default queue


def test_send_task(celery_app):
    """Test 4: Can we send a task?"""
    print("\n" + "=" * 60)
    print("TEST 4: Task Sending")
    print("=" * 60)

    if not celery_app:
        print("❌ Skipping - celery_app not available")
        return False

    target_task = "src.gpu_server.celery_app.gpt_tasks.process_chat_completion"

    print("Attempting to send test task (won't execute without worker)...")
    try:
        result = celery_app.send_task(
            target_task,
            kwargs={
                'task_id': 'test-12345',
                'messages': [{"role": "user", "content": "test"}],
                'model_name': 'openai/gpt-oss-20b',
            },
            task_id='test-12345'
        )
        print(f"✅ Task sent successfully!")
        print(f"   Task ID: {result.id}")
        print(f"   State: {result.state}")
        return True
    except Exception as e:
        print(f"❌ Failed to send task: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("CELERY WORKER CONNECTION TEST")
    print("=" * 60)

    # Run all tests
    celery_app = test_import_celery_app()
    test_task_registration(celery_app)
    test_task_routing(celery_app)
    test_send_task(celery_app)

    print("\n" + "=" * 60)
    print("COMMAND TO START WORKER:")
    print("=" * 60)
    print("\nCORRECT Command (use celery_app module):")
    print("  celery -A src.gpu_server.celery_app.celery_app worker \\")
    print("    -n gpt_worker -Q gpt --concurrency=2 --loglevel=debug")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("If all tests passed, your setup is correct!")
    print("Start the worker with the command above.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

