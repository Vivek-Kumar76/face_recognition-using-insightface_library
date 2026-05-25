"""Pre-download InsightFace weights into the repo (run during Render build)."""
import os
import sys
from pathlib import Path

from insightface.utils import ensure_available

MODEL_NAME = os.environ.get("INSIGHTFACE_MODEL", "buffalo_s")
MODEL_ROOT = Path(__file__).resolve().parent / "insightface_models"
MODEL_ROOT.mkdir(parents=True, exist_ok=True)

pack_dir = MODEL_ROOT / "models" / MODEL_NAME
print(f"Downloading {MODEL_NAME} -> {pack_dir}")
path = ensure_available("models", MODEL_NAME, root=str(MODEL_ROOT))
onnx_files = list(Path(path).glob("*.onnx"))
print(f"Model ready at {path} ({len(onnx_files)} onnx files)")
if not onnx_files:
    print("ERROR: no .onnx files after download", file=sys.stderr)
    sys.exit(1)
