import cv2
import numpy as np
import uuid
import os
import shutil
import pickle
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from insightface.app import FaceAnalysis

app = FastAPI(title="Face Recognition API")

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
#  Load face model ONCE at startup
print("Loading face model... please wait.")
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("Model loaded.")

# Folder to store saved embeddings
EMBEDDINGS_DIR = Path("embeddings")
EMBEDDINGS_DIR.mkdir(exist_ok=True)

TMP_DIR = Path("tmp")
TMP_DIR.mkdir(exist_ok=True)


def save_upload(upload: UploadFile) -> Path:
    """Save an uploaded file to tmp/ and return its path."""
    ext = Path(upload.filename).suffix or ".jpg"
    path = TMP_DIR / f"{uuid.uuid4()}{ext}"
    with open(path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return path


def get_embedding(image_path: Path) -> np.ndarray | None:
    """Detect a face and return its 512-d embedding, or None if no face found."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    faces = face_app.get(img)
    if not faces:
        return None
    # Use the largest detected face (in case of multiple faces) for embedding
    largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return largest.embedding


def cosine_similarity(e1: np.ndarray, e2: np.ndarray) -> float:
    return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))



@app.post("/enroll")
async def enroll(name: str, image: UploadFile = File(...)):
    """
    Save a person's face embedding by name.
    Call this once per person to register them.
    """
    path = save_upload(image)
    try:
        embedding = get_embedding(path)
        if embedding is None:
            raise HTTPException(status_code=400, detail="No face detected in image.")

        save_path = EMBEDDINGS_DIR / f"{name}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(embedding, f)

        return {"status": "enrolled", "name": name}
    finally:
        path.unlink(missing_ok=True)


@app.post("/match")
async def match(
    stored_name: str,
    capture: UploadFile = File(...),
    threshold: float = 0.7
):
    """
    Compare a live capture against an enrolled person.
    Returns matched: true/false and a similarity score.
    """
    embed_path = EMBEDDINGS_DIR / f"{stored_name}.pkl"
    if not embed_path.exists():
        raise HTTPException(status_code=404, detail=f"No enrolled face found for '{stored_name}'. Enroll first.")

    with open(embed_path, "rb") as f:
        stored_embedding = pickle.load(f)

    path = save_upload(capture)
    try:
        live_embedding = get_embedding(path)
        if live_embedding is None:
            raise HTTPException(status_code=400, detail="No face detected in capture image.")

        score = cosine_similarity(stored_embedding, live_embedding)
        matched = score >= threshold

        return {
            "matched": matched,
            "score": round(score, 4),
            "threshold": threshold,
            "name": stored_name,
            "result": "MATCHED " if matched else "NOT MATCHED "
        }
    finally:
        path.unlink(missing_ok=True)


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
        e1 = get_embedding(path1)
        e2 = get_embedding(path2)

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
            "result": "MATCHED " if matched else "NOT MATCHED "
        }
    finally:
        path1.unlink(missing_ok=True)
        path2.unlink(missing_ok=True)


@app.get("/enrolled")
def list_enrolled():
    """List all enrolled people."""
    names = [p.stem for p in EMBEDDINGS_DIR.glob("*.pkl")]
    return {"enrolled": names, "count": len(names)}


@app.delete("/enroll/{name}")
def delete_enrolled(name: str):
    """Remove an enrolled person."""
    path = EMBEDDINGS_DIR / f"{name}.pkl"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"'{name}' not found.")
    path.unlink()
    return {"status": "deleted", "name": name}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)