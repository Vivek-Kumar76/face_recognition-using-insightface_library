"""Pre-download InsightFace weights into the repo (run during Render build)."""
import os
import sys
from pathlib import Path

from insightface.utils import ensure_available

from model_pack import prepare_pack

MODEL_NAME = os.environ.get("INSIGHTFACE_MODEL", "buffalo_s")
MODEL_ROOT = Path(__file__).resolve().parent / "insightface_models"
MODEL_ROOT.mkdir(parents=True, exist_ok=True)

pack_dir = MODEL_ROOT / "models" / MODEL_NAME
print(f"Downloading {MODEL_NAME} -> {pack_dir}")
path = Path(ensure_available("models", MODEL_NAME, root=str(MODEL_ROOT)))
onnx_files = prepare_pack(path, MODEL_NAME)
print(f"Model ready at {path} ({len(onnx_files)} onnx files kept)")
for f in onnx_files:
    print(f"  - {f.name} ({f.stat().st_size // (1024 * 1024)} MB)")
if len(onnx_files) < 2:
    print("ERROR: expected detection + recognition onnx files", file=sys.stderr)
    sys.exit(1)
