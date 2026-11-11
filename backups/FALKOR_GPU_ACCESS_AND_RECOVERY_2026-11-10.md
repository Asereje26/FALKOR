# FALKOR — Punto de recuperación
Fecha: 2026-11-10

## Estado de Colab y GPU
- Google Colab muestra: "Cannot connect to GPU backend — You cannot currently connect to a GPU due to usage limits in Colab."
- T4 GPU estaba seleccionada, pero no se asignó a la sesión.
- Comprobaciones realizadas: `nvidia-smi` no encontrado; PyTorch `2.9.0+cpu`; `torch.cuda.is_available()` devolvió `False`; `/dev/nvidia0` no existe.
- Se eliminó la sesión de ejecución y se intentó crear otra con T4, pero Google Colab volvió a rechazar la conexión por límites de uso.
- No instalar controladores ni sustituir PyTorch para intentar resolver el límite de cuota. Volver a probar más adelante.
- No se ha iniciado el entrenamiento de FALKOR.

## Respaldo persistente existente en Google Drive
Carpeta de respaldo Piper:
`MyDrive/FALKOR_BACKUP/07_COPIAS_DE_SEGURIDAD/FALKOR_RESPALDO_PIPER_AUDIO_20261111_005253`

Ese respaldo contiene Piper y el modelo, además de archivos de configuración/estado y los dos audios de prueba. También existe el ZIP correspondiente en esa ubicación.

Datasets guardados en:
`MyDrive/FALKOR_DATASETS`

Elementos comprobados previamente:
- `acav_train_subset.npy`: forma `(562500, 16, 96)`, tipo `float16`.
- `validation_set_features.npy`: forma `(481345, 96)`, tipo `float32`.
- ESC-50: 2.000 WAV verificados.
- MIT RIR: 270 WAV verificados.

Respaldo de texto anterior en GitHub:
`backups/FALKOR_SESSION_RECOVERY_2026-11-10.md`
Commit conocido: `7168451e89f42f3e165b15660d1349931ecb7a61`.

## Piper y audio
- Piper está respaldado en Drive; en una sesión anterior se recuperó en `/content/PIPER`.
- Archivos comprobados entonces: `/content/PIPER/generate_samples.py` y `/content/PIPER/models/en-us-libritts-high.pt`.
- eSpeak NG quedó instalado en la sesión anterior, y `ctypes.util.find_library("espeak-ng")` devolvió `libespeak-ng.so.1`. La instalación del sistema no persiste al eliminar una sesión de Colab; se puede reinstalar cuando se reanude.
- `webrtcvad==2.0.10` y otros paquetes Python se instalaron/importaron correctamente en la sesión anterior. Las instalaciones Python tampoco deben darse por persistentes después de crear un runtime nuevo.
- El audio de prueba largo `falkor_test_largo.wav` fue escuchado y aprobado.
- Pronunciación de Piper aprobada: usar `length_scales=[1.25]` como punto de partida para las muestras positivas.
- Wake word exacto: `Falkor`, con variantes fonéticas de entrenamiento: `falkor`, `falcor`, `fal core`, `fal kor`.

## Próximos pasos al recuperar acceso
1. Conectar Colab con T4 y verificar `!nvidia-smi`, `torch.cuda.is_available()` y el nombre de la GPU.
2. Montar Google Drive y recuperar Piper desde la carpeta de respaldo indicada arriba.
3. Reinstalar/verificar dependencias del runtime y eSpeak NG.
4. Verificar el generador y el modelo Piper antes de generar muestras.
5. Mantener `length_scales=[1.25]` para la generación positiva aprobada.
6. No iniciar todavía el entrenamiento: primero comprobar y preparar los negativos, revisar rutas/configuración y validar el conjunto de datos.

## Advertencia
Los archivos en `/content` son temporales y se pierden al eliminar el runtime. Los archivos de Google Drive y los respaldos del repositorio son persistentes. No subir pesos grandes, datasets ni audios al repositorio GitHub; conservar esos archivos binarios en Google Drive.
