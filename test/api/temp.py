from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
# Initialize the client (ensure your API key is set in the environment)
client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1",)



# Create a chat completion
response = client.chat.completions.create(
    model="grok-3-mini",
    messages=[
        {"role": "system", "content": "You are a helpful and friendly assistant."},
        {"role": "user", "content": "Write a haiku about the sunrise over the ocean."}
    ],
)



print(response)

# Print the model's reply
print(response.choices[0].message.content)