"""Comprueba que cada pieza configurada en el .env funciona antes de lanzar el agente."""

import asyncio
import os
import sys
from pathlib import Path

import truststore

truststore.inject_into_ssl()

import httpx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent / ".env")

OK = "OK   "
FAIL = "ERROR"


def get(name, default=""):
    return os.getenv(name, default).strip() or default


def check_mic():
    import pyaudio

    pa = pyaudio.PyAudio()
    d = pa.get_default_input_device_info()
    pa.terminate()
    return d["name"]


def check_groq_llm():
    from groq import Groq

    client = Groq(api_key=get("GROQ_API_KEY"))
    r = client.chat.completions.create(
        model=get("LLM_MODEL", "llama-3.3-70b-versatile"),
        max_tokens=30,
        messages=[{"role": "user", "content": "Di 'hola, todo listo' en español y nada más."}],
    )
    return r.choices[0].message.content.strip()


def check_groq_stt():
    r = httpx.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {get('GROQ_API_KEY')}"},
        timeout=20,
    )
    r.raise_for_status()
    ids = [m["id"] for m in r.json()["data"]]
    model = get("STT_MODEL", "whisper-large-v3-turbo")
    if model not in ids:
        raise RuntimeError(f"el modelo {model} no aparece en tu cuenta")
    return f"modelo {model} disponible"


def check_gemini():
    from google import genai

    client = genai.Client(api_key=get("GEMINI_API_KEY"))
    r = client.models.generate_content(
        model=get("LLM_MODEL", "gemini-3.6-flash"),
        contents="Di 'hola, todo listo' en español y nada más.",
    )
    return r.text.strip()


def check_cerebras():
    r = httpx.post(
        "https://api.cerebras.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {get('CEREBRAS_API_KEY')}"},
        json={"model": get("LLM_MODEL", "llama-3.3-70b"), "max_tokens": 30,
              "messages": [{"role": "user", "content": "Di 'hola, todo listo' en español y nada más."}]},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def check_anthropic():
    import anthropic

    client = anthropic.Anthropic(api_key=get("ANTHROPIC_API_KEY"))
    r = client.messages.create(
        model=get("LLM_MODEL", "claude-opus-5"),
        max_tokens=50,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": "Di 'hola, todo listo' en español y nada más."}],
    )
    return "".join(b.text for b in r.content if b.type == "text").strip()


def check_deepgram():
    r = httpx.get(
        "https://api.deepgram.com/v1/projects",
        headers={"Authorization": f"Token {get('DEEPGRAM_API_KEY')}"},
        timeout=20,
    )
    r.raise_for_status()
    return f"{len(r.json().get('projects', []))} proyecto(s)"


def check_edge():
    import edge_tts

    async def run():
        voice = get("EDGE_VOICE", "es-CO-GonzaloNeural")
        c = edge_tts.Communicate("Hola, todo listo.", voice)
        n = 0
        async for chunk in c.stream():
            if chunk["type"] == "audio":
                n += len(chunk["data"])
        return f"voz {voice}, {n} bytes de audio"

    return asyncio.run(run())


def check_kokoro():
    from pipecat.services.kokoro import tts as k

    k._ensure_model_files(k.KOKORO_CACHE_DIR / "kokoro-v1.0.onnx", k.KOKORO_CACHE_DIR / "voices-v1.0.bin")
    return "modelo descargado"


def check_elevenlabs():
    r = httpx.get(
        "https://api.elevenlabs.io/v1/user/subscription",
        headers={"xi-api-key": get("ELEVENLABS_API_KEY")},
        timeout=20,
    )
    r.raise_for_status()
    j = r.json()
    return f"plan {j.get('tier')}, {j.get('character_count')}/{j.get('character_limit')} caracteres usados"


def run(nombre, fn):
    try:
        print(f"{OK} {nombre}: {fn()}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"{FAIL} {nombre}: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    stt = get("STT_PROVIDER", "deepgram").lower()
    llm = get("LLM_PROVIDER", "gemini").lower()
    tts = get("TTS_PROVIDER", "edge").lower()

    checks = [("Micrófono", check_mic)]
    checks.append({"groq": ("Cerebro: Groq", check_groq_llm), "gemini": ("Cerebro: Gemini", check_gemini),
                   "cerebras": ("Cerebro: Cerebras", check_cerebras), "anthropic": ("Cerebro: Anthropic", check_anthropic)}.get(llm, ("Cerebro", lambda: "?")))
    if stt == "groq":
        checks.append(("Transcripción: Groq Whisper", check_groq_stt))
    elif stt == "deepgram":
        checks.append(("Transcripción: Deepgram", check_deepgram))
    else:
        checks.append(("Transcripción: Whisper local", lambda: "se descarga al primer uso"))
    checks.append({"edge": ("Voz: Microsoft Edge", check_edge), "kokoro": ("Voz: Kokoro local", check_kokoro),
                   "elevenlabs": ("Voz: ElevenLabs", check_elevenlabs)}.get(tts, ("Voz", lambda: "?")))

    results = [run(n, f) for n, f in checks]
    print()
    if all(results):
        print("Todo listo. Ejecuta iniciar.bat (o: .venv\\Scripts\\python agent.py)")
    else:
        print("Corrige los errores de arriba en el archivo .env y vuelve a ejecutar verificar.py")
        sys.exit(1)
