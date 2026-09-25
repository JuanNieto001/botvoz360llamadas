"""Lista las voces de tu cuenta de ElevenLabs (propias, clonadas y de la biblioteca).

Uso:
  python voces.py            -> todas tus voces
  python voces.py es         -> solo las que tienen etiqueta de español
  python voces.py prueba ID  -> genera prueba_ID.mp3 con una frase en español para escucharla
"""

import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
if not API_KEY:
    sys.exit("Falta ELEVENLABS_API_KEY en el .env")

HEADERS = {"xi-api-key": API_KEY}
FRASE = (
    "Hola, buenas tardes. Le habla Andrés. Lo llamo porque vi que tiene un plan "
    "con nosotros y quería contarle algo que le puede interesar. ¿Tiene un minuto?"
)


def listar(filtro: str | None):
    r = httpx.get("https://api.elevenlabs.io/v1/voices", headers=HEADERS, timeout=30)
    r.raise_for_status()
    voces = r.json()["voices"]
    print()
    for v in voces:
        labels = v.get("labels") or {}
        etiquetas = " ".join(f"{k}={val}" for k, val in labels.items())
        if filtro and filtro.lower() not in (etiquetas + v["name"]).lower():
            continue
        print(f"{v['voice_id']}  {v['name']:<22} {v.get('category','')[:12]:<12} {etiquetas}")
    print("\nPon el ID elegido en ELEVENLABS_VOICE_ID del .env\n")


def prueba(voice_id: str):
    model = os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5")
    r = httpx.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        headers={**HEADERS, "Content-Type": "application/json"},
        json={"text": FRASE, "model_id": model, "language_code": "es"},
        timeout=60,
    )
    r.raise_for_status()
    out = Path(f"prueba_{voice_id}.mp3")
    out.write_bytes(r.content)
    print(f"Listo: {out}  (ábrelo para escuchar la voz)")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "prueba" and len(args) > 1:
        prueba(args[1])
    else:
        listar(args[0] if args else None)
