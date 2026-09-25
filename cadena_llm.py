"""Cerebro con respaldo: encadena varios proveedores gratis.

Si el primero falla (saturado, sin cupo, caído) o no da la primera palabra en
CADENA_TIMEOUT segundos, se pasa al siguiente EN EL MISMO TURNO, así el cliente
nunca se queda en silencio. Todos hablan el formato de OpenAI, por eso un
solo servicio de Pipecat sirve para todos.

Proveedores (se usan solo los que tengan clave en el .env, en el orden de LLM_CADENA):
  mistral  MISTRAL_API_KEY   gratis, solo pide verificar el celular   https://console.mistral.ai
  nvidia   NVIDIA_API_KEY    gratis con cuenta                        https://build.nvidia.com
  github   GITHUB_TOKEN      gratis con cuenta de GitHub, cupo bajo   https://github.com/settings/tokens
  gemini   GEMINI_API_KEY    gratis, a veces saturado                 https://aistudio.google.com/apikey
"""

import asyncio
import os

from loguru import logger

from pipecat.services.openai.llm import OpenAILLMService

PROVEEDORES = {
    "mistral": ("MISTRAL_API_KEY", "https://api.mistral.ai/v1", "mistral-small-latest", {}),
    "nvidia": ("NVIDIA_API_KEY", "https://integrate.api.nvidia.com/v1", "meta/llama-3.3-70b-instruct", {}),
    "github": ("GITHUB_TOKEN", "https://models.github.ai/inference", "openai/gpt-4.1-mini", {}),
    "gemini": (
        "GEMINI_API_KEY",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "gemini-3.5-flash-lite",
        {"reasoning_effort": "minimal"},
    ),
}


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


class _ConPrimerTrozo:
    """Stream que ya trae leído el primer trozo (para medir que sí respondió)."""

    def __init__(self, primero, it, stream):
        self._primero = primero
        self._it = it
        self._stream = stream

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        yield self._primero
        async for chunk in self._it:
            yield chunk

    async def close(self):
        close = getattr(self._stream, "close", None)
        if close:
            await close()


class CadenaLLMService(OpenAILLMService):
    def __init__(self, orden: list[str], timeout_secs: float = 4.0, **kwargs):
        backends = []
        for nombre in orden:
            if nombre not in PROVEEDORES:
                logger.warning(f"Proveedor desconocido en LLM_CADENA: {nombre}")
                continue
            var, url, modelo, extra = PROVEEDORES[nombre]
            clave = _env(var)
            if not clave:
                continue
            if nombre == "gemini":
                # Varios modelos de Gemini con la misma clave: si uno está
                # saturado, otro suele responder.
                modelos = _env("GEMINI_MODELOS") or "gemini-3.5-flash-lite,gemini-3.6-flash,gemini-3.1-flash-lite"
                for m in [x.strip() for x in modelos.split(",") if x.strip()]:
                    nivel = "low" if m.startswith(("gemini-3.7", "gemini-3.8")) else "minimal"
                    backends.append((f"gemini:{m}", clave, url, m, {"reasoning_effort": nivel}))
            elif nombre == "mistral":
                # Los modelos Ministral tienen cupo en el plan gratis; mistral-small a veces no.
                modelos = _env("MISTRAL_MODELOS") or "ministral-14b-latest,ministral-8b-latest"
                for m in [x.strip() for x in modelos.split(",") if x.strip()]:
                    backends.append((f"mistral:{m}", clave, url, m, {}))
            else:
                modelo = _env(f"{nombre.upper()}_MODEL") or modelo
                backends.append((nombre, clave, url, modelo, extra))
        if not backends:
            raise SystemExit(
                "LLM_CADENA no tiene ningún proveedor con clave. Pon al menos una de: "
                + ", ".join(v[0] for v in PROVEEDORES.values())
            )

        nombre, clave, url, modelo, _ = backends[0]
        settings = kwargs.pop("settings", None) or OpenAILLMService.Settings()
        settings.model = modelo
        super().__init__(api_key=clave, base_url=url, settings=settings, **kwargs)
        self._timeout = timeout_secs
        self._backends = [(n, self.create_client(api_key=k, base_url=u), m, e) for n, k, u, m, e in backends]
        logger.info("Cerebro en cadena: " + " -> ".join(f"{n} ({m})" for n, _, m, _ in self._backends))

    async def get_chat_completions(self, context):
        adapter = self.get_llm_adapter()
        params_from_context = adapter.get_llm_invocation_params(
            context,
            system_instruction=self._settings.system_instruction,
            convert_developer_to_user=True,
        )
        base = self.build_chat_completion_params(params_from_context)

        ultimo_error = None
        for i, (nombre, client, modelo, extra) in enumerate(self._backends):
            params = {**base, "model": modelo}
            if extra:
                params["extra_body"] = {**params.get("extra_body", {}), **extra}
            try:
                # El último de la cadena tiene más paciencia: es la última opción.
                limite = self._timeout if i < len(self._backends) - 1 else max(self._timeout, 12.0)
                stream = await asyncio.wait_for(client.chat.completions.create(**params), timeout=limite)
                it = stream.__aiter__()
                primero = await asyncio.wait_for(it.__anext__(), timeout=limite)
                if i > 0:
                    logger.warning(f"[cerebro] respondió el respaldo: {nombre}")
                return _ConPrimerTrozo(primero, it, stream)
            except Exception as e:  # noqa: BLE001
                ultimo_error = e
                motivo = "tardó demasiado" if isinstance(e, TimeoutError) else f"{type(e).__name__}: {str(e)[:120]}"
                logger.warning(f"[cerebro] {nombre} falló ({motivo}); probando el siguiente")
        raise RuntimeError(f"Ningún cerebro respondió. Último error: {ultimo_error}")
