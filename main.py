import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

FEATHERLESS_API_KEY = os.getenv("FEATHERLESS_API_KEY")
if not FEATHERLESS_API_KEY:
    print(
        "Missing FEATHERLESS_API_KEY. Add it to a .env file in the project root.",
        file=sys.stderr,
    )
    sys.exit(1)

response = requests.post(
    "https://api.featherless.ai/v1/chat/completions",
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {FEATHERLESS_API_KEY}",
    },
    json={
        "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! How are you?"},
        ],
    },
    timeout=120,
)

if not response.ok:
    print(response.status_code, response.text, file=sys.stderr)
    sys.exit(1)

data = response.json()
print(data["choices"][0]["message"]["content"])
