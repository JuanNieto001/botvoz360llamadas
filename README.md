# Agente de voz conversacional (100 % gratis)

Hablas por el micrófono y el agente te responde con una voz humana en español, manteniendo la conversación el tiempo que quieras. Puedes interrumpirlo mientras habla, como en una llamada real.

Es el paso 1 del proyecto de asesor de callcenter. Los pasos 2 (entrenar con ventas reales) y 3 (clonar una voz) se montan encima sin cambiar la base.

## Qué usa (configuración gratuita, probada contra el firewall de la oficina)

| Pieza | Servicio | Costo | Clave |
|---|---|---|---|
| Orquestación | [Pipecat](https://github.com/pipecat-ai/pipecat), open source | Gratis | No |
| Cerebro | Gemini Flash (Google AI Studio) | Gratis, sin tarjeta | Sí |
| Voz a texto | Deepgram nova-3 en español | 200 USD de crédito gratis, sin tarjeta | Sí |
| Texto a voz | Voces neuronales de Microsoft Edge (es-CO-GonzaloNeural) | Gratis | No |
| Detección de voz | Silero VAD + Smart Turn v3, corren en tu PC | Gratis | No |

Groq, OpenRouter, Cartesia y HuggingFace están **bloqueados por el firewall** de la oficina (Fortinet), por eso no son la opción por defecto aunque también sean gratis. Cerebras (gratis) sí pasa y queda como alternativa de cerebro.

## Puesta en marcha (5 minutos)

1. Clave de Gemini: entra a https://aistudio.google.com/apikey con tu cuenta de Google y pulsa "Create API key".
2. Clave de Deepgram: regístrate en https://console.deepgram.com y en "API Keys" crea una.
3. Copia `.env.example` como `.env` y pega las dos claves.
4. Abre una terminal en esta carpeta y comprueba que todo responde:
   ```
   .venv\Scripts\python verificar.py
   ```
5. Doble clic en `iniciar.bat` (o `.venv\Scripts\python agent.py`). El agente saluda primero. Ctrl+C para salir.

Cada conversación queda guardada en `conversaciones/` con hora y quién dijo qué. Eso luego sirve como material de entrenamiento.

## Ajustes útiles (archivo .env)

- `EDGE_VOICE`: `es-CO-GonzaloNeural` (hombre) o `es-CO-SalomeNeural` (mujer). También hay voces de México, Argentina y EE. UU.
- `EDGE_RATE` y `EDGE_PITCH`: `-10%` suena más pausado, `-20Hz` más grave.
- `PERSONA`: archivo de `prompts/` que define cómo habla. `conversador` es charla libre; `asesor_ventas` es el borrador para el paso 2.
- `LLM_MODEL`: vacío usa el Gemini Flash más reciente. `modelos.py` lista los disponibles con tu clave.
- `VAD_CONFIDENCE`: si hay ruido de fondo y el agente se interrumpe solo, súbelo a 0.8.
- `VAD_STOP_SECS`: si te corta a mitad de frase, súbelo a 0.4.

## Si algo falla

- "Falta la variable X": el `.env` está incompleto.
- Error de certificado SSL: ya está resuelto en el código con `truststore` (la red de la oficina intercepta las conexiones). Si aparece en otro script tuyo, añade `import truststore; truststore.inject_into_ssl()` al inicio.
- Una página HTML de Fortinet en el error: ese servicio está bloqueado en la red de la oficina. Cambia de proveedor en el `.env` o prueba desde otra red.
- El agente no me escucha: ejecuta `dispositivos.py` y pon el índice correcto en `MIC_DEVICE_INDEX`.
- Gemini responde "429" o "quota": el plan gratis tiene un tope por minuto. Espera un momento.
- Para ver todo el detalle técnico: `LOG_LEVEL=DEBUG`.

## Límites del plan gratis

- Gemini gratis: usa `gemini-3.5-flash-lite`, que tiene cupo amplio. Los modelos Flash más nuevos (3.6, 3.7, 3.8) solo dan 5 respuestas por minuto y 20 al día gratis, se agotan en una charla.
- Deepgram: 200 USD ≈ más de 400 horas de transcripción.
- Edge TTS: sin límite publicado, pero es un servicio no oficial de Microsoft; si un día deja de funcionar, cambia `TTS_PROVIDER=kokoro` (voz local ya descargada, algo lenta en este PC) mientras vuelve.

## Siguientes pasos del proyecto

1. **Entrenar al asesor**: reunir grabaciones o transcripciones de las mejores ventas, extraer guion, objeciones y respuestas, y meterlas en `prompts/asesor_ventas.md`.
2. **Clonar la voz gratis**: este PC no tiene tarjeta gráfica, así que clonar en local no es viable en tiempo real. Opciones sin costo: (a) correr [Chatterbox Multilingual](https://github.com/resemble-ai/chatterbox) (MIT, clona con 10 s de audio, español) en Google Colab gratis y exponerlo como servidor; (b) el plan de 5 USD de ElevenLabs, que ya incluye clonación instantánea, si algún día hay presupuesto.
3. **Llamadas reales**: cambiar el transporte de micrófono local por Twilio usando el ejemplo `twilio-chatbot` de Pipecat. El resto no cambia.

## Clonar la forma de hablar de un asesor

1. Copia las grabaciones de las llamadas donde vendió en la carpeta `llamadas/` (mp3, wav, m4a, ogg y otros). También sirven transcripciones en `.txt` con líneas tipo `Asesor: texto`.
2. Ejecuta:
   ```
   .venv\Scripts\python entrenar.py
   ```
   Transcribe cada llamada, identifica al asesor, extrae sus frases textuales, cómo abre, qué pregunta, cómo responde cada objeción y cómo cierra, y lo junta todo en `prompts/asesor_clonado.md`.
3. Revisa ese archivo, sobre todo los precios y condiciones del producto.
4. En el `.env` pon `PERSONA=asesor_clonado` y abre `iniciar.bat`.

Si lo cortas o se agota el cupo gratis de Gemini, vuelve a ejecutarlo: sigue donde iba y no repite lo ya procesado. Las transcripciones quedan legibles, con ASESOR y CLIENTE, en `llamadas/_procesado/transcripciones/`.

Costo: la transcripción gasta del crédito gratis de Deepgram, unos 0,26 USD por hora de audio. Con 200 USD alcanza para cientos de horas de llamadas. El análisis con Gemini es gratis.

Importante: las llamadas tienen datos personales de clientes. Se envían a Deepgram y a Google para procesarlas, y el plan gratis de Gemini puede usar lo que recibe para mejorar sus productos. Confirma con la empresa que está autorizado antes de usar grabaciones reales. El script ya borra números de tarjeta y quita los nombres de clientes de la personalidad final.
