# FALKOR — checkpoint 2026-11-11

## Estado verificado
- Proyecto: FALKOR.
- Wake word: Falkor (no "Hey Falkor").
- Voz de muestra: en-us.
- Piper phonemize API disponible como `piper_phonemize`.
- Adaptador local `/content/PIPER/espeak_phonemizer.py` creado y probado.
- Resultado de prueba: `Falkor` -> `fˈɔlkɚ`.
- Los seis símbolos del resultado están presentes en `phoneme_id_map` del modelo.
- `/content/PIPER/generate_samples.py` importa correctamente después de añadir `/content/PIPER` al `sys.path`.
- `generate_samples.py` no fue modificado.
- El modelo LibriTTS High y los datasets están disponibles en el entorno Colab/Drive según las comprobaciones previas.
- Entrenamiento NO iniciado.
- Generación de muestras de entrenamiento NO iniciada.
- Distribución final de los 2,000 audios totales pendiente de aprobación.

## Adaptador probado
El archivo `espeak_phonemizer.py` incluido en este checkpoint convierte la lista de fonemas de `piper_phonemize.phonemize_espeak` en la cadena que espera el generador.

## Límites de este respaldo GitHub
Este checkpoint guarda el adaptador y el estado técnico. No incluye modelos binarios, datasets, audios, credenciales ni archivos grandes de entrenamiento. Esos archivos deben permanecer en Google Drive/Colab.
