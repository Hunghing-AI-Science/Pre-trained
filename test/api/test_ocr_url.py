"""
Test OCR API with internet image URLs

This test demonstrates how to use the OCR API with images from the internet.

Usage:
    python test/api/test_ocr_url.py
"""

import requests
import time
import base64


class OCRAPIClient:
    """Simple client for testing OCR API with URLs"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def download_and_encode_image(self, image_url: str) -> str:
        """
        Download image from URL and encode to base64 data URI

        Args:
            image_url: URL of the image to download

        Returns:
            Base64 data URI string
        """
        print(f"\n📥 Downloading image from URL: {image_url}")

        response = requests.get(image_url, timeout=30)
        response.raise_for_status()

        image_bytes = response.content
        print(f"   Downloaded: {len(image_bytes)} bytes")

        # Encode to base64
        base64_encoded = base64.b64encode(image_bytes).decode('utf-8')

        # Detect image format from Content-Type header
        content_type = response.headers.get('Content-Type', 'image/jpeg')
        if 'png' in content_type:
            image_format = 'png'
        elif 'gif' in content_type:
            image_format = 'gif'
        elif 'webp' in content_type:
            image_format = 'webp'
        else:
            image_format = 'jpeg'

        data_uri = f"data:image/{image_format};base64,{base64_encoded}"
        print(f"   Encoded to base64: {len(data_uri)} chars")

        return data_uri

    def ocr_from_url(self, image_url: str, prompt: str = "Free OCR."):
        """
        Create OCR task from image URL

        This method downloads the image and converts it to base64,
        then sends it to the API.

        Args:
            image_url: URL of the image to process
            prompt: OCR prompt

        Returns:
            Task ID
        """
        # Step 1: Download and encode image
        image_data_uri = self.download_and_encode_image(image_url)

        # Step 2: Send to API
        url = f"{self.base_url}/v1/ocr/completions"

        payload = {
            "model": "deepseek-ocr",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": image_data_uri}
                    ]
                }
            ]
        }

        print(f"\n📤 Creating OCR task")
        print(f"   Prompt: {prompt}")

        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()

        result = response.json()
        task_id = result["id"]

        print(f"\n✅ Task created: {task_id}")
        print(f"   Status: {result['status']}")

        return task_id

    def get_task_status(self, task_id: str):
        """Get task status and result"""
        url = f"{self.base_url}/v1/ocr/tasks/{task_id}"

        response = requests.get(url)
        response.raise_for_status()

        return response.json()

    def wait_for_completion(self, task_id: str, timeout: int = 60, interval: int = 2):
        """
        Wait for task to complete

        Args:
            task_id: Task ID
            timeout: Maximum time to wait in seconds
            interval: Poll interval in seconds

        Returns:
            Task result
        """
        print(f"\n⏳ Waiting for task {task_id} to complete...")

        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")

            result = self.get_task_status(task_id)
            status = result["status"]

            print(f"   Status: {status}", end="\r")

            if status == "completed":
                print(f"\n✅ Task completed!")
                return result
            elif status == "failed":
                error = result.get("error", "Unknown error")
                raise Exception(f"Task failed: {error}")

            time.sleep(interval)

    def _print_curl_command(self, image_url: str, prompt: str):
        """Print equivalent curl command"""
        print(f"\n🔧 Equivalent curl commands:")
        print("=" * 80)
        print("# Step 1: Download and encode image to base64")
        print(f"curl -s '{image_url}' | base64 > /tmp/image.b64")
        print("\n# Step 2: Send to OCR API")
        print(f"""curl -X POST '{self.base_url}/v1/ocr/completions' \\
  -H 'Content-Type: application/json' \\
  -d '{{
    "model": "deepseek-ocr",
    "messages": [
      {{
        "role": "user",
        "content": [
          {{"type": "text", "text": "{prompt}"}},
          {{"type": "image", "image": "data:image/jpeg;base64,'$(cat /tmp/image.b64)'"}}
        ]
      }}
    ]
  }}'""")
        print("\n# Or in one line (simpler):")
        print(f"""IMAGE_BASE64=$(curl -s '{image_url}' | base64 | tr -d '\\n')
curl -X POST '{self.base_url}/v1/ocr/completions' \\
  -H 'Content-Type: application/json' \\
  -d '{{"model":"deepseek-ocr","messages":[{{"role":"user","content":[{{"type":"text","text":"{prompt}"}},{{"type":"image","image":"data:image/jpeg;base64,'"$IMAGE_BASE64"'"}}]}}]}}'""")
        print("=" * 80)


def test_ocr_from_internet():
    """Test OCR with an image from the internet"""
    print("=" * 80)
    print("TEST: OCR from Internet Image URL")
    print("=" * 80)

    client = OCRAPIClient()

    # Example image URL - you can replace this with any image URL
    image_url = "https://lh5.googleusercontent.com/proxy/kuKDfpDNa6qnFEdbSwdtH-mlfR2FLNC_XR0gijFad0P5qa1pgjPOk2-pZVjP4SlRNNL_waT3cAD5NzONat2DoeDW7Uiw"
    prompt = "Extract all text from this image."

    # Print curl command
    client._print_curl_command(image_url, prompt)

    try:
        # Create task
        task_id = client.ocr_from_url(image_url, prompt)

        # Wait for completion
        result = client.wait_for_completion(task_id)

        # Display results
        print("\n📝 OCR Result:")
        print("=" * 80)

        if result.get("result"):
            ocr_result = result["result"]
            print(f"Text: {ocr_result['choices'][0]['text']}")
            print(f"\n📊 Usage:")
            print(f"   Prompt tokens: {ocr_result['usage']['prompt_tokens']}")
            print(f"   Completion tokens: {ocr_result['usage']['completion_tokens']}")
            print(f"   Total tokens: {ocr_result['usage']['total_tokens']}")

        print("=" * 80)
        print("\n✅ Test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_urls():
    """Test OCR with multiple different image URLs"""
    print("\n" + "=" * 80)
    print("TEST: Multiple Image URLs")
    print("=" * 80)

    client = OCRAPIClient()

    # Example image URLs
    test_images = [
        {
            "url": "https://via.placeholder.com/600x200.png?text=Hello+World",
            "prompt": "Extract text"
        },
        {
            "url": "https://via.placeholder.com/600x200.png?text=Sample+Document",
            "prompt": "Read this document"
        },
    ]

    results = []

    for idx, test_case in enumerate(test_images, 1):
        print(f"\n--- Test Image {idx} ---")

        try:
            task_id = client.ocr_from_url(test_case["url"], test_case["prompt"])
            result = client.wait_for_completion(task_id, timeout=30)

            if result.get("result"):
                text = result["result"]["choices"][0]["text"]
                print(f"\n✅ Extracted: {text[:100]}...")
                results.append(True)
            else:
                print(f"\n⚠️  No result yet")
                results.append(False)

        except Exception as e:
            print(f"\n❌ Failed: {str(e)}")
            results.append(False)

    # Summary
    passed = sum(results)
    total = len(results)

    print("\n" + "=" * 80)
    print(f"Results: {passed}/{total} images processed successfully")
    print("=" * 80)

    return all(results)


def test_error_handling():
    """Test error handling with invalid URLs"""
    print("\n" + "=" * 80)
    print("TEST: Error Handling")
    print("=" * 80)

    client = OCRAPIClient()

    # Test with invalid URL
    print("\n--- Invalid URL ---")
    invalid_url = "https://invalid-domain-12345.com/image.jpg"

    try:
        task_id = client.ocr_from_url(invalid_url, "Extract text")
        result = client.wait_for_completion(task_id, timeout=10)

        if result.get("status") == "failed":
            print(f"\n✅ Correctly handled invalid URL")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            return True
        else:
            print(f"\n⚠️  Expected failure but got: {result.get('status')}")
            return False

    except Exception as e:
        print(f"\n✅ Correctly raised exception: {str(e)}")
        return True


def test_health_check():
    """Test API health"""
    print("=" * 80)
    print("TEST: Health Check")
    print("=" * 80)

    try:
        response = requests.get("http://localhost:8000/")
        response.raise_for_status()

        result = response.json()
        print(f"\n✅ API is running")
        print(f"   Service: {result.get('service')}")
        print(f"   Status: {result.get('status')}")

        return True

    except Exception as e:
        print(f"\n❌ API is not running: {str(e)}")
        print(f"\n💡 Make sure to start the API server:")
        print(f"   uvicorn src.gpu_server.api:app --reload")
        return False


def main():
    """Run all tests"""
    print("=" * 80)
    print("OCR API URL Test Suite")
    print("=" * 80)
    print("\nTesting OCR API with Internet Image URLs")
    print("Base URL: http://localhost:8000")
    print("=" * 80)

    # Track results
    results = []

    # Health check first
    if not test_health_check():
        print("\n⚠️  API server is not running. Please start it first.")
        return

    # Run tests
    results.append(("OCR from Internet", test_ocr_from_internet()))
    results.append(("Multiple URLs", test_multiple_urls()))
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

