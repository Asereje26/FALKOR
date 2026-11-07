# FALKOR — checkpoint de respaldo

Fecha: 2026-11-06

## Estado confirmado en la sesión de Google Colab

- `/content/piper-sample-generator`: existe.
- Modelo Piper `models/en-us-libritts-high.pt`: existe.
- Archivo JSON del modelo: existe.
- Adaptador `/content/piper-sample-generator/generate_samples.py`: existe en Colab; se ha respaldado una copia en `tools/backup/generate_samples.py`.
- `/content/my_model.yaml`: existe en Colab; falta copiar su contenido actual a GitHub.
- `/content/openwakeword`: existe en Colab.
- `/content/falkor_output/falkor`: carpeta existe; todavía no se ha confirmado que contenga muestras o características de entrenamiento.
- `/content/mit_rirs`: no existe actualmente en Colab.
- `/content/fma_wav`: no existe actualmente en Colab.
- `/content/acav_train_subset.npy`: no existe actualmente en Colab.
- `/content/validation_set_features.npy`: no existe actualmente en Colab.

## Entrenamiento

La ejecución de augment falló porque `train.py` intenta importar `generate_samples` como módulo superior:

```python
from generate_samples import generate_samples
```

El paquete instalado expone la función desde:

```python
from piper_sample_generator.__main__ import generate_samples
```

Se creó un adaptador de compatibilidad que reenvía la llamada al generador oficial, elimina el argumento no compatible `auto_reduce_batch_size` y fija la ruta al modelo Piper existente. El adaptador se respaldó en GitHub.

## Dependencias comprobadas en la sesión anterior

Las importaciones de `torchinfo`, `torchmetrics`, `pronouncing`, `audiomentations`, `torch_audiomentations` y `speechbrain` funcionaron. La advertencia TF32 de PyTorch no era un error fatal.

## Próximos pasos seguros

1. Respaldar el contenido actual de `my_model.yaml` en GitHub.
2. Respaldar el notebook actualizado y cualquier parche local de `openwakeword/train.py`.
3. Verificar el contenido de `falkor_output/falkor` antes de continuar.
4. Recuperar los conjuntos de datos que faltan.
5. Reanudar augment y entrenamiento solo después de comprobar los respaldos.

## Limitaciones

Los archivos temporales de Colab no se copian automáticamente a GitHub. Los modelos binarios y conjuntos de datos grandes pueden exceder los límites prácticos de GitHub; no subirlos al repositorio sin revisar tamaño y conveniencia. Este checkpoint registra el estado conocido, no es una copia de los datos que faltan.
