"""
    main()
if __name__ == "__main__":


        sys.exit(1)
        print(f"Error: {e}")
    except Exception as e:
        
        print("="*80)
        print(ocr_text)
        print("="*80)
        print("OCR RESULT:")
        print("\n" + "="*80)
        
        ocr_text = client.process_image(image_path, prompt)
        # Process image
    try:
    
    client = OCRClient()
    # Create client
    
        sys.exit(1)
        print(f"Error: Image file not found: {image_path}")
    if not Path(image_path).exists():
    
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Free OCR."
    image_path = sys.argv[1]
    
        sys.exit(1)
        print("  python test_client.py document.png 'Convert to markdown'")
        print("  python test_client.py document.png")
        print("\nExample:")
        print("Usage: python test_client.py <image_path> [prompt]")
    if len(sys.argv) < 2:
    """Example usage"""
def main():


        return ocr_text
        
        print(f"Tokens used: {usage['total_tokens']} (prompt: {usage['prompt_tokens']}, completion: {usage['completion_tokens']})")
        print(f"\nOCR completed!")
        
        usage = result["result"]["usage"]
        ocr_text = result["result"]["choices"][0]["text"]
        
        result = self.wait_for_completion(task_id)
        print("Waiting for completion...")
        
        print(f"Task created: {task_id}")
        task_id = self.create_ocr_task(image_path, prompt)
        print(f"Submitting image: {image_path}")
        """
        This is a convenience method that handles the full workflow

        Process an image and return the OCR text
        """
    def process_image(self, image_path: str, prompt: str = "Free OCR.") -> str:
    
            time.sleep(poll_interval)
            print(f"Task status: {status}... waiting {poll_interval}s")
            
                raise Exception(f"Task failed: {error}")
                error = status_data.get("error", "Unknown error")
            elif status == "failed":
                return status_data
            if status == "completed":
            
            status = status_data["status"]
            status_data = self.get_task_status(task_id)
            
                raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
            if time.time() - start_time > timeout:
        while True:
        
        start_time = time.time()
        """
            The completed task result
        Returns:

            timeout: Maximum seconds to wait
            poll_interval: Seconds between status checks
            task_id: The task ID
        Args:

        Wait for a task to complete
        """
    def wait_for_completion(self, task_id: str, poll_interval: int = 2, timeout: int = 300) -> dict:
    
        return response.json()
        response.raise_for_status()
        response = requests.get(url)
        url = f"{self.base_url}/v1/ocr/tasks/{task_id}"
        """Get the status of a task"""
    def get_task_status(self, task_id: str) -> dict:
    
            return result["id"]
            result = response.json()
            
            response.raise_for_status()
            response = requests.post(url, files=files, data=data)
            
            }
                "model": "deepseek-ocr"
                "prompt": prompt,
            data = {
            files = {"file": f}
        with open(image_path, "rb") as f:
        
        url = f"{self.base_url}/v1/ocr/completions"
        """
            task_id: The ID of the created task
        Returns:

        Submit an image for OCR processing
        """
    def create_ocr_task(self, image_path: str, prompt: str = "Free OCR.") -> str:
    
        self.base_url = base_url
    def __init__(self, base_url: str = "http://localhost:8000"):
class OCRClient:


from pathlib import Path
import sys
import time
import requests
"""
Example client for testing the DeepSeek OCR API

