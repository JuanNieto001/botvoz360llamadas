"""Lista los modelos de Gemini disponibles con tu clave (para elegir LLM_MODEL)."""

import os
from pathlib import Path

import truststore

truststore.inject_into_ssl()

from dotenv import load_dotenv  # noqa: E402
from google import genai  # noqa: E402

load_dotenv(Path(__file__).resolve().parent / ".env")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", "").strip())

print()
for m in client.models.list():
    name = m.name.replace("models/", "")
    if "flash" in name or "pro" in name:
        print(f"  {name}")
print("\nPon el elegido en LLM_MODEL del .env (los 'flash' son los rápidos, ideales para voz)\n")
