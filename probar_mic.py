"""Medidor de nivel del micrófono. Habla y mira si la barra se mueve.

Uso:  .venv\\Scripts\\python probar_mic.py         (micrófono por defecto)
      .venv\\Scripts\\python probar_mic.py 5       (micrófono con índice 5, ver dispositivos.py)
"""

import sys

import numpy as np
import pyaudio

idx = int(sys.argv[1]) if len(sys.argv) > 1 else None
pa = pyaudio.PyAudio()
info = pa.get_device_info_by_index(idx) if idx is not None else pa.get_default_input_device_info()
print(f"\nMicrófono: [{info['index']}] {info['name']}")
print("Habla durante 10 segundos. Silencio = barra vacía; voz normal = barra a más de la mitad.\n")

stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True,
                 frames_per_buffer=1600, input_device_index=idx)
peak_total = 0
for _ in range(100):
    audio = np.frombuffer(stream.read(1600, exception_on_overflow=False), dtype=np.int16)
    peak = int(np.abs(audio).max())
    peak_total = max(peak_total, peak)
    bar = "#" * min(50, peak // 200)
    print(f"\r{bar:<50} {peak:>6}", end="", flush=True)
stream.close()
pa.terminate()

print("\n")
if peak_total < 300:
    print("NO se detecta voz. Revisa: botón de silencio (mute) de la diadema, nivel del micrófono en")
    print("Configuración de Windows > Sonido > Entrada, y que la diadema sea el dispositivo de entrada.")
elif peak_total < 1500:
    print("Se escucha muy bajo. Sube el nivel del micrófono en Windows o acércate más.")
else:
    print("Micrófono OK. Ya puedes ejecutar iniciar.bat")
