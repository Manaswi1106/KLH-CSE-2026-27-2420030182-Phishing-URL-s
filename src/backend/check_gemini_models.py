import os
from google import genai

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

print("Available Gemini models:\n")

for model in client.models.list():
    if "generateContent" in str(model.supported_actions):
        print(model.name)