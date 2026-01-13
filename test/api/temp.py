from openai import OpenAI
from dotenv import load_dotenv
import os
def request_to_curl(prep):
    curl = ["curl -X", prep.method, repr(prep.url)]
    for k, v in prep.headers.items():
        curl.append(f"-H {repr(f'{k}: {v}')}")
    if prep.body:
        curl.append(f"-d {repr(prep.body.decode() if isinstance(prep.body, bytes) else prep.body)}")
    return " \\\n  ".join(curl)

load_dotenv()
# Initialize the client (ensure your API key is set in the environment)
client = OpenAI(base_url="http://localhost:8000/v1",)



# Create a chat completion
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "system", "content": "You are a helpful and friendly assistant."},
        {"role": "user", "content": "Write a python scipt that can find the prime number."}
    ],
)



print(response)

# Print the model's reply
print(response.choices[0].message.content)