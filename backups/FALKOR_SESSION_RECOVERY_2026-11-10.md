# FALKOR — session recovery record

**Recorded:** 2026-11-10  
**Repository:** `Asereje26/FALKOR`  
**Purpose:** Record the verified state of the Piper / wake-word preparation session and point to the Drive backup.

## Verified backups

- Colab notebook copy saved in Google Drive as `FALKOR_RESPALDO_SESION_20261110` (user confirmed the save).
- Google Drive archive: `/content/drive/MyDrive/FALKOR_RESPALDO_SESION_20261111_002351.zip`
- Drive backup folder: `/content/drive/MyDrive/FALKOR_RESPALDO_SESION_20261111_002351`
- Archive size reported by Colab: 182.21 MB.
- ZIP integrity test: passed (`testzip()` returned no error).
- 101 files reported in the archive.

## Verified technical state

- Piper model file used: `/content/PIPER/models/en-us-libritts-high.pt`
- The checkpoint loaded successfully with `torch.load(..., map_location="cpu", weights_only=True)`.
- Loaded object type: `piper_train.vits.models.SynthesizerTrn`.
- PyTorch 2.6 safe loading protection remained enabled; the session did **not** switch to `weights_only=False`.
- The following classes/functions were selectively allowlisted after checking their identities:
  - Piper: `WN`, `Log`, `ConvFlow`, `PosteriorEncoder`, `Flip`, `ResidualCouplingLayer`, `DDSConv`, `ElementwiseAffine`, `ResBlock2`, `LayerNorm`, `ResidualCouplingBlock`, `Generator`, `StochasticDurationPredictor`, `FFN`
  - PyTorch: `ConvTranspose1d`, `torch.nn.utils.weight_norm.WeightNorm`
  - Python built-in: `builtins.getattr`

## Not yet verified / not completed

- Successful WAV generation for the wake phrase **“Falkor”** has not yet been confirmed.
- The 20,000-step training run has not started.
- ONNX export and wake-word validation have not been completed.
- Negative audio datasets still need to be prepared/verified.
- This GitHub record intentionally does not include the 182 MB Piper checkpoint or large datasets. Those remain in Google Drive to avoid committing large binary files.

## Recovery notes

1. Open the saved Colab notebook copy in Drive.
2. Confirm Drive is mounted and the archive above is present.
3. Restore the Piper/configuration folders from the ZIP if the Colab runtime has reset.
4. Reinstall dependencies using the backed-up `dependencias_instaladas.txt` as a reference, checking compatibility before changing PyTorch.
5. Reapply the selective PyTorch safe-global allowlist only after verifying the corresponding installed Piper/PyTorch classes.
6. First test generating a single WAV containing “Falkor”; do not begin full training until generation, data paths, negatives, and validation inputs are verified.

**Important:** This record describes the state reported by the Colab session. It is not a claim that the Piper audio generation or model training has succeeded.
