import asyncio
import logging
import numpy as np
import uuid
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from insightface.app import FaceAnalysis

from model_pack import prepare_pack

# buffalo_s on Render: prune unused onnx (landmarks etc.) so only det+rec load into RAM.
MODEL_NAME = os.getenv("INSIGHTFACE_MODEL", "buffalo_s")
DET_SIZE = int(os.getenv("INSIGHTFACE_DET_SIZE", "320"))
# Project-local models (NOT under .gitignore — Render must ship build artifacts).
MODEL_ROOT = Path(__file__).resolve().parent / "insightface_models"

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup so the first API call does not hit Render's request timeout."""
    logger.info("Preloading face model at startup...")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, load_model)
    logger.info("Face model ready.")
    yield


app = FastAPI(title="Face Recognition API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "model_loaded": face_app is not None,
    }

# Global model variable (lazy loaded)
face_app = None


def model_pack_dir() -> Path:
    return MODEL_ROOT / "models" / MODEL_NAME


def ensure_model_pack() -> Path:
    """Download model pack if missing (e.g. local dev); verify onnx files exist."""
    from insightface.utils import ensure_available

    pack_dir = model_pack_dir()
    onnx_files = prepare_pack(pack_dir, MODEL_NAME) if pack_dir.is_dir() else []
    if len(onnx_files) < 2:
        logger.warning("Model pack missing at %s — downloading %s", pack_dir, MODEL_NAME)
        ensure_available("models", MODEL_NAME, root=str(MODEL_ROOT))
        onnx_files = prepare_pack(pack_dir, MODEL_NAME)
    if len(onnx_files) < 2:
        raise RuntimeError(
            f"Need detection + recognition .onnx in {pack_dir}. "
            "Run: python download_model.py"
        )
    logger.info("Model pack OK: %s (%s)", pack_dir, [p.name for p in onnx_files])
    return pack_dir


def load_model():
    """Lazy load face model only when needed."""
    global face_app
    if face_app is None:
        pack_dir = model_pack_dir()
        logger.info("Loading face model %s (det=%s) from %s", MODEL_NAME, DET_SIZE, pack_dir)
        try:
            ensure_model_pack()
            face_app = FaceAnalysis(
                name=MODEL_NAME,
                root=str(MODEL_ROOT),
                providers=["CPUExecutionProvider"],
                allowed_modules=["detection", "recognition"],
            )
            if "detection" not in face_app.models:
                raise RuntimeError(
                    f"Detection model not loaded. Found tasks: {list(face_app.models)}. "
                    f"Onnx in {pack_dir}: {[p.name for p in pack_dir.glob('*.onnx')]}"
                )
            face_app.prepare(ctx_id=0, det_size=(DET_SIZE, DET_SIZE))
        except HTTPException:
            raise
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc!r}" if not str(exc) else f"{type(exc).__name__}: {exc}"
            logger.exception("Failed to load face model")
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Face model '{MODEL_NAME}' failed to load ({err}). "
                    f"Expected files under {pack_dir}. "
                    "On Render: redeploy and confirm build runs download_model.py successfully."
                ),
            ) from exc
        logger.info("Face model loaded.")
    return face_app


TMP_DIR = Path("tmp")
TMP_DIR.mkdir(exist_ok=True)


def save_upload(upload: UploadFile) -> Path:
    """Save an uploaded file to tmp/ and return its path."""
    ext = Path(upload.filename).suffix or ".jpg"
    path = TMP_DIR / f"{uuid.uuid4()}{ext}"
    with open(path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return path


async def run_embedding(image_path: Path) -> np.ndarray | None:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, get_embedding, image_path)


def get_embedding(image_path: Path) -> np.ndarray | None:
    """Detect a face and return its 512-d embedding, or None if no face found."""
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        return None
    model = load_model()
    faces = model.get(img)
    if not faces:
        return None
    # Use the largest detected face (in case of multiple faces) for embedding
    largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return largest.embedding


def cosine_similarity(e1: np.ndarray, e2: np.ndarray) -> float:
    return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))


@app.post("/match-two")
async def match_two(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    threshold: float = 0.7
):
    """
    Directly compare two uploaded images without enrolling.
    """
    path1 = save_upload(image1)
    path2 = save_upload(image2)
    try:
        e1 = await run_embedding(path1)
        e2 = await run_embedding(path2)

        if e1 is None:
            raise HTTPException(status_code=400, detail="No face detected in image 1.")
        if e2 is None:
            raise HTTPException(status_code=400, detail="No face detected in image 2.")

        score = cosine_similarity(e1, e2)
        matched = score >= threshold

        return {
            "matched": matched,
            "score": round(score, 4),
            "threshold": threshold,
        }
    finally:
        path1.unlink(missing_ok=True)
        path2.unlink(missing_ok=True)




