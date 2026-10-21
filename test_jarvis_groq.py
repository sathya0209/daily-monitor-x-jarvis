from pathlib import Path
from dotenv import load_dotenv
import os
from groq import Groq

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
api_key = os.getenv("GROQ_API_KEY", "").strip()

print("Project folder:", BASE_DIR)
print("Model configured:", model)
print("API key present:", bool(api_key))
print("API key length:", len(api_key) if api_key else 0)

if not api_key:
    print("\nDIAGNOSIS: GROQ_API_KEY was not found in the .env file next to this script.")
    raise SystemExit(1)

try:
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Reply naturally in one short sentence."},
            {"role": "user", "content": "Say hello to me."},
        ],
        max_completion_tokens=100,
    )
    print("\nGROQ CONNECTION: OK")
    print("Model reply:", response.choices[0].message.content)
except Exception as exc:
    print("\nGROQ CONNECTION: FAILED")
    print("Error type:", type(exc).__name__)
    print("Error details:", str(exc))
    print("\nDo not share your API key. You can share only the error type and details above.")
