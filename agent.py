"""Agente de voz conversacional (paso 1: conversación estable por micrófono).

Pipeline: micrófono -> voz a texto -> modelo de lenguaje -> texto a voz -> parlantes.

Cada pieza se elige en el archivo .env (ver .env.example). Configuración
gratuita por defecto: Gemini (cerebro) + Deepgram (transcripción) + voz de Microsoft Edge.
"""

import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# La red de esta oficina intercepta las conexiones seguras con un certificado
# propio. truststore hace que Python confíe en los certificados de Windows.
import truststore

truststore.inject_into_ssl()

from dotenv import load_dotenv  # noqa: E402
from loguru import logger  # noqa: E402

from pipecat.audio.vad.silero import SileroVADAnalyzer  # noqa: E402
from pipecat.audio.vad.vad_analyzer import VADParams  # noqa: E402
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams  # noqa: E402
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3  # noqa: E402
from pipecat.turns.user_start.min_words_user_turn_start_strategy import MinWordsUserTurnStartStrategy  # noqa: E402
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import TurnAnalyzerUserTurnStopStrategy  # noqa: E402
from pipecat.turns.user_turn_strategies import UserTurnStrategies  # noqa: E402
from pipecat.frames.frames import (  # noqa: E402
    Frame,
    LLMFullResponseEndFrame,
    LLMRunFrame,
    TranscriptionFrame,
    TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.workers.runner import WorkerRunner  # noqa: E402
from pipecat.pipeline.worker import PipelineParams, PipelineWorker  # noqa: E402
from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: E402
from pipecat.processors.aggregators.llm_response_universal import (  # noqa: E402
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor  # noqa: E402
from pipecat.observers.base_observer import BaseObserver, FramePushed  # noqa: E402
from pipecat.frames.frames import (  # noqa: E402
    BotStartedSpeakingFrame,
    ErrorFrame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    TextFrame as _TextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.transcriptions.language import Language  # noqa: E402
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

logger.remove()
logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"))


# ----------------------------------------------------------------- utilidades
def env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def env_int(name: str) -> int | None:
    value = env(name)
    return int(value) if value is not None else None


def env_float(name: str, default: float) -> float:
    value = env(name)
    return float(value) if value is not None else default


def require(name: str) -> str:
    value = env(name)
    if not value:
        logger.error(f"Falta la variable {name} en el archivo .env")
        sys.exit(1)
    return value


def load_persona() -> str:
    """Lee el prompt del sistema desde prompts/<PERSONA>.md."""
    persona = env("PERSONA", "conversador")
    path = BASE_DIR / "prompts" / f"{persona}.md"
    if not path.exists():
        logger.error(f"No existe el archivo de persona {path}")
        sys.exit(1)
    return path.read_text(encoding="utf-8").strip()


# --------------------------------------------------------- registro de la charla
class TranscriptWriter:
    """Archivo de la conversación actual, compartido por los dos "taps"."""

    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._file = open(log_dir / f"{stamp}.txt", "a", encoding="utf-8")

    def write(self, who: str, text: str):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {who}: {text}"
        print(line, flush=True)
        self._file.write(line + "\n")
        self._file.flush()


class TranscriptTap(FrameProcessor):
    """Muestra en consola y guarda lo que dice cada uno.

    Se pone uno justo después del STT (lo que dices tú, porque el agregador de
    usuario consume esos frames y no los reenvía) y otro después del TTS (lo
    que dice el agente).
    """

    def __init__(self, writer: TranscriptWriter):
        super().__init__()
        self._writer = writer
        self._bot_buffer: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            self._writer.write("TÚ", frame.text)
        elif isinstance(frame, TTSTextFrame):
            self._bot_buffer.append(frame.text)
        elif isinstance(frame, LLMFullResponseEndFrame) and self._bot_buffer:
            self._writer.write("AGENTE", " ".join(self._bot_buffer).strip())
            self._bot_buffer = []
        await self.push_frame(frame, direction)


class TimingObserver(BaseObserver):
    """Anota en consola cuánto tarda cada parte de la respuesta.

    [tiempos] silencio->turno: cuánto esperó para decidir que terminaste
              turno->cerebro:  cuánto tardó Gemini en dar la primera palabra
              cerebro->voz:    cuánto tardó la voz en empezar a sonar
    """

    def __init__(self):
        super().__init__()
        self._t_vad = self._t_turn = self._t_llm = None
        self._llm_first = False
        self._seen: set[int] = set()

    async def on_push_frame(self, data: FramePushed):
        f = data.frame
        if id(f) in self._seen:
            return
        self._seen.add(id(f))
        if len(self._seen) > 5000:
            self._seen.clear()
        now = time.monotonic()
        if isinstance(f, VADUserStoppedSpeakingFrame):
            self._t_vad = now
        elif isinstance(f, UserStartedSpeakingFrame):
            logger.info("[tiempos] cliente empezó a hablar")
        elif isinstance(f, InterruptionFrame):
            logger.info("[tiempos] el cliente interrumpió al agente")
        elif isinstance(f, UserStoppedSpeakingFrame):
            self._t_turn = now
            w = f"{now - self._t_vad:.2f}s" if self._t_vad else "?"
            logger.info(f"[tiempos] turno del cliente terminado (silencio->turno {w})")
        elif isinstance(f, LLMFullResponseStartFrame):
            self._t_llm = now
            self._llm_first = True
        elif isinstance(f, _TextFrame) and self._llm_first and self._t_llm and type(f).__name__ == "LLMTextFrame":
            self._llm_first = False
            base = self._t_turn or self._t_llm
            logger.info(f"[tiempos] primera palabra del cerebro (turno->cerebro {now - base:.2f}s)")
            self._t_llm = now
        elif isinstance(f, BotStartedSpeakingFrame):
            parts = []
            if self._t_llm:
                parts.append(f"cerebro->voz {now - self._t_llm:.2f}s")
            if self._t_vad:
                parts.append(f"TOTAL desde que callaste {now - self._t_vad:.2f}s")
            if parts:
                logger.info(f"[tiempos] agente empezó a hablar ({', '.join(parts)})")
            self._t_vad = self._t_turn = self._t_llm = None
        elif isinstance(f, ErrorFrame):
            logger.warning(f"[tiempos] error: {f.error[:200]}")


# ------------------------------------------------------------------ proveedores
def build_stt():
    """Voz a texto. STT_PROVIDER = deepgram (crédito gratis) | groq (gratis, bloqueado en la oficina) | whisper (local)."""
    provider = env("STT_PROVIDER", "deepgram").lower()
    language = env("STT_LANGUAGE", "es")

    if provider == "deepgram":
        from pipecat.services.deepgram.stt import DeepgramSTTService

        return DeepgramSTTService(
            api_key=require("DEEPGRAM_API_KEY"),
            settings=DeepgramSTTService.Settings(
                model=env("STT_MODEL", "nova-3-general"),
                language=language,
                smart_format=True,
                punctuate=True,
                interim_results=True,
            ),
        )

    if provider == "whisper":
        from pipecat.services.whisper.stt import WhisperSTTService

        return WhisperSTTService(
            model=env("STT_MODEL", "small"),
            device="cpu",
            compute_type="int8",
            settings=WhisperSTTService.Settings(language=Language(language)),
        )

    from pipecat.services.groq.stt import GroqSTTService

    return GroqSTTService(
        api_key=require("GROQ_API_KEY"),
        settings=GroqSTTService.Settings(
            model=env("STT_MODEL", "whisper-large-v3-turbo"),
            language=Language(language),
            prompt="Transcripción de una llamada telefónica en español.",
        ),
    )


def build_llm(system_prompt: str):
    """Cerebro. LLM_PROVIDER = gemini (gratis) | cerebras (gratis) | anthropic (de pago) | groq (gratis, bloqueado en la oficina)."""
    provider = env("LLM_PROVIDER", "cadena").lower()
    max_tokens = int(env("LLM_MAX_TOKENS", "300"))

    if provider == "anthropic":
        from pipecat.services.anthropic.llm import AnthropicLLMService

        return AnthropicLLMService(
            api_key=require("ANTHROPIC_API_KEY"),
            settings=AnthropicLLMService.Settings(
                model=env("LLM_MODEL", "claude-opus-5"),
                system_instruction=system_prompt,
                max_tokens=max_tokens,
                enable_prompt_caching=True,
                extra={"output_config": {"effort": env("LLM_EFFORT", "low")}},
            ),
        )

    if provider == "cadena":
        from cadena_llm import CadenaLLMService

        return CadenaLLMService(
            orden=[x.strip() for x in env("LLM_CADENA", "mistral,nvidia,github,gemini").split(",") if x.strip()],
            timeout_secs=env_float("CADENA_TIMEOUT", 4.0),
            settings=CadenaLLMService.Settings(
                system_instruction=system_prompt,
                max_tokens=max_tokens,
                temperature=env_float("LLM_TEMPERATURE", 0.7),
            ),
        )

    if provider == "gemini":
        from pipecat.services.google.llm import GoogleLLMService

        return GoogleLLMService(
            api_key=require("GEMINI_API_KEY"),
            # Si el primer trozo tarda más de 5 s, reintenta una vez.
            retry_on_timeout=True,
            retry_timeout_secs=5.0,
            settings=GoogleLLMService.Settings(
                # Sin LLM_MODEL se usa el Flash más reciente que trae Pipecat.
                **({"model": env("LLM_MODEL")} if env("LLM_MODEL") else {}),
                system_instruction=system_prompt,
                max_tokens=max_tokens,
                temperature=env_float("LLM_TEMPERATURE", 0.7),
                # Pensamiento al mínimo: con el nivel por defecto Gemini tarda
                # más de 10 s en empezar a hablar; con "minimal", menos de 1 s.
                thinking=GoogleLLMService.ThinkingConfig(
                    thinking_level=env("GEMINI_THINKING", "minimal")
                ),
            ),
        )

    if provider == "cerebras":
        from pipecat.services.cerebras.llm import CerebrasLLMService

        return CerebrasLLMService(
            api_key=require("CEREBRAS_API_KEY"),
            settings=CerebrasLLMService.Settings(
                model=env("LLM_MODEL", "llama-3.3-70b"),
                system_instruction=system_prompt,
                max_tokens=max_tokens,
                temperature=env_float("LLM_TEMPERATURE", 0.7),
            ),
        )

    from pipecat.services.groq.llm import GroqLLMService

    return GroqLLMService(
        api_key=require("GROQ_API_KEY"),
        settings=GroqLLMService.Settings(
            model=env("LLM_MODEL", "llama-3.3-70b-versatile"),
            system_instruction=system_prompt,
            max_tokens=max_tokens,
            temperature=env_float("LLM_TEMPERATURE", 0.7),
        ),
    )


def build_tts():
    """Texto a voz. TTS_PROVIDER = edge (gratis) | kokoro (local) | elevenlabs | cartesia."""
    provider = env("TTS_PROVIDER", "edge").lower()
    language = env("TTS_LANGUAGE", "es")

    if provider == "kokoro":
        from pipecat.services.kokoro.tts import KokoroTTSService

        return KokoroTTSService(
            settings=KokoroTTSService.Settings(
                voice=env("KOKORO_VOICE", "em_santa"),
                language=Language.ES,
                speed=env_float("TTS_SPEED", 1.0),
            )
        )

    if provider == "cartesia":
        from pipecat.services.cartesia.tts import CartesiaTTSService

        return CartesiaTTSService(
            api_key=require("CARTESIA_API_KEY"),
            settings=CartesiaTTSService.Settings(
                voice=require("CARTESIA_VOICE_ID"),
                model=env("CARTESIA_MODEL", "sonic-3.6"),
                language=language,
            ),
        )

    if provider == "elevenlabs":
        from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

        return ElevenLabsTTSService(
            api_key=require("ELEVENLABS_API_KEY"),
            settings=ElevenLabsTTSService.Settings(
                voice=require("ELEVENLABS_VOICE_ID"),
                model=env("ELEVENLABS_MODEL", "eleven_flash_v2_5"),
                language=language,
                stability=env_float("ELEVENLABS_STABILITY", 0.5),
                similarity_boost=env_float("ELEVENLABS_SIMILARITY", 0.8),
                use_speaker_boost=True,
            ),
        )

    from edge_tts_service import EdgeTTSService

    return EdgeTTSService(
        voice=env("EDGE_VOICE", "es-CO-GonzaloNeural"),
        rate=env("EDGE_RATE", "+0%"),
        pitch=env("EDGE_PITCH", "+0Hz"),
    )


# ----------------------------------------------------------------------- main
async def main():
    system_prompt = load_persona()

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            input_device_index=env_int("MIC_DEVICE_INDEX"),
            output_device_index=env_int("SPEAKER_DEVICE_INDEX"),
        )
    )

    stt = build_stt()
    llm = build_llm(system_prompt)
    tts = build_tts()

    # El prompt del sistema va en la configuración del LLM (system_instruction);
    # el contexto solo arranca con la instrucción de saludar.
    context = LLMContext(
        messages=[
            {
                "role": "user",
                "content": "(La llamada acaba de conectar. Empieza como lo harías normalmente "
                "al inicio de una llamada, breve y natural, en una o dos frases.)",
            },
        ]
    )

    # Detección de voz (Silero) + detección de fin de turno (Smart Turn v3,
    # entrenado en español). Permite interrumpir al agente y evita que te
    # corte a mitad de frase. La transcripción por segmentos (Groq/Whisper)
    # también depende de estos eventos de voz.
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    confidence=env_float("VAD_CONFIDENCE", 0.7),
                    start_secs=env_float("VAD_START_SECS", 0.2),
                    stop_secs=env_float("VAD_STOP_SECS", 0.2),
                )
            ),
            user_turn_strategies=UserTurnStrategies(
                # El agente solo se calla si el cliente dice al menos N palabras;
                # así un ruido o una voz de fondo no lo corta a mitad de frase.
                start=[MinWordsUserTurnStartStrategy(min_words=int(env("MIN_PALABRAS_INTERRUMPIR", "2")))],
                # Smart Turn decide si terminaste de hablar. Si duda, espera como
                # máximo TURNO_MAX_SILENCIO segundos (antes eran 3).
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3(
                            params=SmartTurnParams(stop_secs=env_float("TURNO_MAX_SILENCIO", 1.0))
                        )
                    )
                ],
            ),
        ),
    )

    writer = TranscriptWriter(BASE_DIR / "conversaciones")

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            TranscriptTap(writer),
            aggregators.user(),
            llm,
            tts,
            TranscriptTap(writer),
            transport.output(),
            aggregators.assistant(),
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=None,
        observers=[TimingObserver()],
    )

    print(
        f"\n=== Agente listo (STT={env('STT_PROVIDER', 'deepgram')}, LLM={env('LLM_PROVIDER', 'cadena')}, "
        f"TTS={env('TTS_PROVIDER', 'edge')}). Habla por el micrófono. Ctrl+C para salir. ===\n",
        flush=True,
    )
    await worker.queue_frames([LLMRunFrame()])

    runner = WorkerRunner(handle_sigint=True)
    await runner.add_workers(worker)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
