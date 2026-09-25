"""Servicio de texto a voz para Pipecat usando las voces neuronales de Microsoft Edge.

Gratis y sin clave. Voces en español recomendadas:
  es-CO-GonzaloNeural (hombre, Colombia)   es-CO-SalomeNeural (mujer, Colombia)
  es-MX-JorgeNeural   (hombre, México)     es-MX-DaliaNeural  (mujer, México)
  es-AR-TomasNeural   (hombre, Argentina)  es-US-AlonsoNeural (hombre, EE. UU.)

Edge entrega MP3 por trozos; aquí se decodifica en streaming a PCM de 16 bits
con PyAV, así el primer audio suena antes de que termine la frase completa.
"""

import re
from collections.abc import AsyncGenerator

import av
import edge_tts
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService


# Emojis y símbolos que la voz leería en voz alta ("cara sonriente").
_EMOJI = re.compile("[🀀-🫿☀-➿️‍*#_~]+")


class EdgeTTSService(TTSService):
    def __init__(self, *, voice: str = "es-CO-GonzaloNeural", rate: str = "+0%", pitch: str = "+0Hz", **kwargs):
        kwargs.setdefault("sample_rate", 24000)
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            settings=TTSSettings(model="edge", voice=voice, language="es"),
            **kwargs,
        )
        self._rate = rate
        self._pitch = pitch

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        voice = self._settings.voice
        text = re.sub(r"\[[^\]]*\]", "", _EMOJI.sub("", text)).strip()
        if not text:
            return
        logger.debug(f"Edge TTS [{voice}]: {text}")
        await self.start_ttfb_metrics()
        await self.start_tts_usage_metrics(text)

        codec = av.CodecContext.create("mp3", "r")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=self.sample_rate)

        def to_pcm(packets) -> bytes:
            out = bytearray()
            for packet in packets:
                for frame in codec.decode(packet):
                    for rf in resampler.resample(frame):
                        out += rf.to_ndarray().tobytes()
            return bytes(out)

        try:
            communicate = edge_tts.Communicate(text, voice, rate=self._rate, pitch=self._pitch)
            first = True
            async for chunk in communicate.stream():
                if chunk["type"] != "audio":
                    continue
                pcm = to_pcm(codec.parse(chunk["data"]))
                if not pcm:
                    continue
                if first:
                    first = False
                    await self.stop_ttfb_metrics()
                yield TTSAudioRawFrame(audio=pcm, sample_rate=self.sample_rate, num_channels=1, context_id=context_id)

            # Vaciar lo que quede en el decodificador.
            tail = to_pcm(codec.parse(b"")) + to_pcm([None])
            if tail:
                yield TTSAudioRawFrame(audio=tail, sample_rate=self.sample_rate, num_channels=1, context_id=context_id)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Edge TTS error: {e}")
            yield ErrorFrame(error=f"Edge TTS: {e}")
        finally:
            await self.stop_ttfb_metrics()
