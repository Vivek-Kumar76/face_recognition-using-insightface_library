"""InsightFace model pack layout helpers (build + runtime)."""
import shutil
from pathlib import Path

# FaceAnalysis still loads every .onnx in the folder before filtering by allowed_modules.
# Remove unused files so buffalo_s fits Render 512MB RAM (~15MB vs ~158MB on disk).
PACK_KEEP_FILES: dict[str, set[str]] = {
    "buffalo_s": {"det_500m.onnx", "w600k_mbf.onnx"},
    "buffalo_sc": {"det_500m.onnx", "w600k_mbf.onnx"},
    "buffalo_m": {"det_2.5g.onnx", "w600k_r50.onnx"},
    "buffalo_l": {"det_10g.onnx", "w600k_r50.onnx"},
}


def flatten_pack(pack_dir: Path) -> list[Path]:
    """Move nested .onnx files into pack root."""
    pack_dir.mkdir(parents=True, exist_ok=True)
    for onnx in list(pack_dir.rglob("*.onnx")):
        if onnx.parent != pack_dir:
            dest = pack_dir / onnx.name
            if not dest.exists():
                shutil.move(str(onnx), str(dest))
    return list(pack_dir.glob("*.onnx"))


def prune_pack(pack_dir: Path, model_name: str) -> list[Path]:
    """Delete onnx files not needed for detection + recognition only."""
    keep = PACK_KEEP_FILES.get(model_name)
    if not keep:
        return list(pack_dir.glob("*.onnx"))
    for onnx in list(pack_dir.glob("*.onnx")):
        if onnx.name not in keep:
            print(f"Pruning unused model file: {onnx.name}")
            onnx.unlink(missing_ok=True)
    return list(pack_dir.glob("*.onnx"))


def prepare_pack(pack_dir: Path, model_name: str) -> list[Path]:
    """Flatten zip layout and keep only det + recognition weights."""
    flatten_pack(pack_dir)
    onnx_files = prune_pack(pack_dir, model_name)
    return onnx_files
