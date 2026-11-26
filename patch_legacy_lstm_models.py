"""
Patch legacy TensorFlow/Keras HDF5 model configs that contain
LSTM(time_major=False), which is rejected by newer Keras releases.

Run from the project root:
    python patch_legacy_lstm_models.py

The script:
- Scans models/<SYMBOL>/model.h5 and model.keras.
- Only edits files with an HDF5 signature.
- Removes time_major only when its value is False from serialized LSTM configs.
- Creates a .before_time_major_fix.bak backup before editing.
- Attempts to load each patched model with compile=False.
- Does not change model weights, scalers, or metadata.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"


def is_hdf5(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(8) == HDF5_SIGNATURE
    except OSError:
        return False


def decode_model_config(raw):
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    elif hasattr(raw, "decode"):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raw = str(raw)
    return json.loads(raw)


def remove_obsolete_lstm_flag(node, location="root"):
    """Remove only LSTM time_major=False; leave True untouched and report it."""
    changed = 0
    unsupported_true = []

    if isinstance(node, dict):
        class_name = node.get("class_name")
        cfg = node.get("config")

        if class_name in {"LSTM", "CuDNNLSTM"} and isinstance(cfg, dict):
            if cfg.get("time_major") is False:
                del cfg["time_major"]
                changed += 1
                print(f"    Removed obsolete time_major=False at {location}")
            elif cfg.get("time_major") is True:
                unsupported_true.append(location)

        for key, value in list(node.items()):
            sub_changed, sub_true = remove_obsolete_lstm_flag(
                value, f"{location}.{key}"
            )
            changed += sub_changed
            unsupported_true.extend(sub_true)

    elif isinstance(node, list):
        for index, value in enumerate(node):
            sub_changed, sub_true = remove_obsolete_lstm_flag(
                value, f"{location}[{index}]"
            )
            changed += sub_changed
            unsupported_true.extend(sub_true)

    return changed, unsupported_true


def patch_hdf5_model(path: Path) -> tuple[bool, str]:
    if not is_hdf5(path):
        return False, "not an HDF5 model file"

    try:
        import h5py
        import numpy as np
    except Exception as exc:
        return False, f"Cannot import h5py/numpy: {exc}"

    try:
        with h5py.File(path, "r") as h5:
            if "model_config" not in h5.attrs:
                return False, "HDF5 file has no full-model model_config attribute"
            raw = h5.attrs["model_config"]
            config = decode_model_config(raw)

        changed, unsupported_true = remove_obsolete_lstm_flag(config)
        if unsupported_true:
            return (
                False,
                "found time_major=True; left unchanged because removing it "
                "could change sequence semantics: " + ", ".join(unsupported_true),
            )
        if changed == 0:
            return False, "no obsolete LSTM time_major=False setting found"

        backup = path.with_name(path.name + ".before_time_major_fix.bak")
        if not backup.exists():
            shutil.copy2(path, backup)
            print(f"    Backup created: {backup.name}")

        encoded = json.dumps(config, separators=(",", ":")).encode("utf-8")
        with h5py.File(path, "r+") as h5:
            del h5.attrs["model_config"]
            h5.attrs.create("model_config", np.bytes_(encoded))

        return True, f"patched {changed} LSTM config(s); backup: {backup.name}"

    except Exception as exc:
        return False, f"patch failed: {type(exc).__name__}: {exc}"


def main():
    if not MODELS_DIR.is_dir():
        print(f"Models directory not found: {MODELS_DIR}")
        return 1

    model_files = sorted(
        list(MODELS_DIR.glob("*/model.h5"))
        + list(MODELS_DIR.glob("*/model.keras"))
    )
    if not model_files:
        print(f"No model.h5 or model.keras files found in {MODELS_DIR}")
        return 1

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Models found: {len(model_files)}")

    patched = []
    skipped = []
    failed = []

    for model_path in model_files:
        print(f"\n[{model_path.parent.name}] {model_path.name}")

        if not is_hdf5(model_path):
            skipped.append((model_path, "not HDF5; likely native .keras ZIP format"))
            print("    Skipped: not an HDF5 file.")
            continue

        ok, message = patch_hdf5_model(model_path)
        if ok:
            patched.append(model_path)
            print(f"    PATCHED: {message}")
        else:
            skipped.append((model_path, message))
            print(f"    Not patched: {message}")

    print("\nVerifying HDF5 models with the installed Keras...")
    try:
        from tensorflow.keras.models import load_model
    except Exception as exc:
        print(f"Could not import TensorFlow Keras: {exc}")
        return 1

    for model_path in model_files:
        if not is_hdf5(model_path):
            continue
        try:
            loaded = load_model(str(model_path), compile=False)
            input_shape = getattr(loaded, "input_shape", None)
            del loaded
            print(f"    [LOAD OK] {model_path.parent.name}: input_shape={input_shape}")
        except Exception as exc:
            failed.append((model_path, f"{type(exc).__name__}: {exc}"))
            print(f"    [LOAD FAILED] {model_path.parent.name}: {exc}")

    print("\n" + "=" * 68)
    print("PATCH SUMMARY")
    print("=" * 68)
    print(f"Files patched: {len(patched)}")
    print(f"Files skipped: {len(skipped)}")
    print(f"Files still failing to load: {len(failed)}")

    if failed:
        print("\nRemaining load errors:")
        for path, error in failed:
            print(f"  {path.parent.name}: {error}")
        print("\nKeep the .bak backups and send these remaining errors.")
        return 1

    print("\nAll detected HDF5 models loaded successfully, or were not HDF5 files.")
    print("Restart Flask and test /api/predict/BTCUSDT and /api/predict/BNBUSDT.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
