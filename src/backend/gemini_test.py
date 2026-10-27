import os
from google import genai

print("API key found:", bool(os.environ.get("GEMINI_API_KEY")))

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY")
)

response = client.interactions.create(
    model="gemini-3.6-flash",
    input="Reply with exactly: Gemini connection successful."
)

print(response.output_text)