"""Settings for the PediCXR backend. Every value can be changed with an environment variable."""
import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BACKEND_DIR.parent
CXR_CAP_DIR = REPO_DIR / "cxr_cap"          # the trained model code (model.py, dataset.py, ...)

STORAGE_DIR = Path(os.getenv("PEDICXR_STORAGE", BACKEND_DIR / "storage"))
UPLOAD_DIR = STORAGE_DIR / "uploads"         # X-rays as uploaded + a PNG copy for the viewer
HEATMAP_DIR = STORAGE_DIR / "heatmaps"       # Grad-CAM heatmaps (transparent PNG)
DB_DIR = STORAGE_DIR / "db"                  # kept apart so it is never served as a file

DATABASE_URL = os.getenv("PEDICXR_DB_URL", f"sqlite:///{(DB_DIR / 'pedicxr.db').as_posix()}")

MODEL_PATH = Path(os.getenv("PEDICXR_MODEL", CXR_CAP_DIR / "fold0_best.pt"))
MODEL_NAME = os.getenv("PEDICXR_MODEL_NAME", "resnet50-fold0")
RESULTS_DIR = CXR_CAP_DIR / "results"        # saved test results shown on the Model info page
THRESHOLD = float(os.getenv("PEDICXR_THRESHOLD", "0.5"))
IMG_SIZE = 512                               # model input size (same as training)
MAX_SIDE = 1024                              # big uploads are shrunk to this, like the training PNGs

MAX_UPLOAD_MB = int(os.getenv("PEDICXR_MAX_UPLOAD_MB", "25"))
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm", ".dicom"}

CORS_ORIGINS = [o.strip() for o in os.getenv(
    "PEDICXR_CORS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]


def ensure_dirs() -> None:
    for d in (UPLOAD_DIR, HEATMAP_DIR, DB_DIR):
        d.mkdir(parents=True, exist_ok=True)
