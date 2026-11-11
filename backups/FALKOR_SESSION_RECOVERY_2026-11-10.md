# FALKOR — session recovery record

**Updated:** 2026-11-10  
**Repository:** `Asereje26/FALKOR`  
**Purpose:** Record the verified state of the Piper / wake-word preparation session and point to the Google Drive backup.

## Previously verified backups

- Colab notebook copy saved in Google Drive as `FALKOR_RESPALDO_SESION_20261110` (user confirmed the save).
- Previous Google Drive archive: `/content/drive/MyDrive/FALKOR_RESPALDO_SESION_20261111_002351.zip`
- Previous Drive backup folder: `/content/drive/MyDrive/FALKOR_RESPALDO_SESION_20261111_002351`
- Previous archive size: 182.21 MB; 101 files; ZIP integrity test passed.
- That earlier archive predates the successful Piper WAV test and therefore does not contain the new audio samples.

## Verified technical state

- Piper model: `/content/PIPER/models/en-us-libritts-high.pt`
- Checkpoint loaded successfully with `torch.load(..., map_location="cpu", weights_only=True)`.
- Loaded object type: `piper_train.vits.models.SynthesizerTrn`.
- PyTorch 2.6 safe-loading protection remained enabled; no `weights_only=False` bypass was used.
- Selective safe-global allowlist was checked and applied for the required Piper/PyTorch classes and the built-in `getattr`.
- eSpeak NG was installed successfully; `ctypes.util.find_library("espeak-ng")` returned `libespeak-ng.so.1`.
- Piper generated `/content/FALKOR_PIPER_TEST/falkor_test.wav` successfully (12,684 bytes).
- Piper generated `/content/FALKOR_PIPER_TEST/falkor_test_largo.wav` successfully using `length_scales=[1.25]` (18,444 bytes).
- User listened to and approved the pronunciation and longer duration of the second sample.
- Approved positive-sample starting parameter: `length_scales=[1.25]`.
- Wake phrase remains **“Falkor”**, with the agreed phonetic training variants: `falkor`, `falcor`, `fal core`, and `fal kor`.

## Not yet completed

- The new WAV samples have not yet been confirmed as copied into the refreshed Google Drive archive.
- The latest Colab notebook copy has not yet been confirmed saved under a new timestamped name.
- The 20,000-step training run has not started.
- ONNX export and wake-word validation have not been completed.
- Negative audio datasets still need to be prepared/verified.

## Recovery notes

1. Open the saved Colab notebook copy in Drive and confirm the latest cells are saved.
2. Copy the two test WAV files into the Drive backup folder and create a new timestamped ZIP; verify its integrity.
3. If the Colab runtime resets, restore the Piper/configuration files from the previous verified ZIP and reinstall dependencies, including eSpeak NG.
4. Reapply the selective PyTorch safe-global allowlist only after verifying the corresponding installed Piper/PyTorch classes.
5. Before full training, prepare and verify positive and negative samples, dataset paths, ACAV features, and validation inputs.

**Important:** This record documents the confirmed audio-generation milestone. It is not a claim that the new Drive archive has already been created or that model training has started.
