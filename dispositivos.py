"""Lista los micrófonos y parlantes disponibles con su índice.

Copia el índice que quieras usar en MIC_DEVICE_INDEX / SPEAKER_DEVICE_INDEX del .env.
"""

import pyaudio

pa = pyaudio.PyAudio()
default_in = pa.get_default_input_device_info()["index"]
default_out = pa.get_default_output_device_info()["index"]

print("\nMICRÓFONOS (entrada):")
for i in range(pa.get_device_count()):
    d = pa.get_device_info_by_index(i)
    if d["maxInputChannels"] > 0:
        mark = "  <- por defecto" if i == default_in else ""
        print(f"  [{i}] {d['name']}{mark}")

print("\nPARLANTES (salida):")
for i in range(pa.get_device_count()):
    d = pa.get_device_info_by_index(i)
    if d["maxOutputChannels"] > 0:
        mark = "  <- por defecto" if i == default_out else ""
        print(f"  [{i}] {d['name']}{mark}")

pa.terminate()
print()
