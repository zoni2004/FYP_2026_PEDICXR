"""Saving uploaded X-rays and heatmaps to disk."""
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .. import config


class UploadError(Exception):
    """A problem with the uploaded file that the user can fix."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class SavedUpload:
    stem: str
    upload_file: str      # the file exactly as uploaded
    image_file: str       # grayscale PNG copy the viewer shows
    gray: np.ndarray      # uint8 (H, W), what the model sees


def add_cxr_cap_to_path() -> None:
    p = str(config.CXR_CAP_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def _write_png(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"could not encode {path.name}")
    path.write_bytes(buf.tobytes())


def _decode(data: bytes, ext: str, raw_path: Path) -> np.ndarray:
    if ext in (".dcm", ".dicom"):
        add_cxr_cap_to_path()
        from prepare_data import dicom_to_uint8  # same DICOM handling as training
        try:
            g = dicom_to_uint8(raw_path)
        except Exception as e:
            raise UploadError("This DICOM file could not be read. Check that it is a chest X-ray DICOM.") from e
    else:
        g = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if g is None:
            raise UploadError("This image could not be read. Upload a valid PNG, JPG or DICOM file.")
    if g.ndim == 3:
        if g.shape[-1] in (3, 4):
            g = cv2.cvtColor(g, cv2.COLOR_BGRA2GRAY if g.shape[-1] == 4 else cv2.COLOR_BGR2GRAY)
        else:  # multi-frame DICOM: use the first frame
            g = g[0]
    if g.dtype != np.uint8:  # e.g. 16-bit PNG
        g = g.astype(np.float32)
        lo, hi = float(g.min()), float(g.max())
        g = ((g - lo) / max(hi - lo, 1e-6) * 255).astype(np.uint8)
    return g


def _shrink(g: np.ndarray) -> np.ndarray:
    h, w = g.shape
    s = config.MAX_SIDE / max(h, w)
    if s < 1:
        g = cv2.resize(g, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return g


def save_upload(filename: str, data: bytes) -> SavedUpload:
    ext = Path(filename or "").suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise UploadError("Unsupported file type. Upload a PNG, JPG or DICOM (.dcm) chest X-ray.", 415)
    if not data:
        raise UploadError("The file is empty.")
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise UploadError(f"The file is larger than {config.MAX_UPLOAD_MB} MB.", 413)

    stem = uuid.uuid4().hex
    raw = config.UPLOAD_DIR / f"{stem}_original{ext}"
    raw.write_bytes(data)
    try:
        gray = _shrink(_decode(data, ext, raw))
        if min(gray.shape) < 64:
            raise UploadError("The image is too small. Upload the full-size X-ray.")
        image_file = f"{stem}.png"
        _write_png(config.UPLOAD_DIR / image_file, gray)
    except Exception:
        raw.unlink(missing_ok=True)
        raise
    return SavedUpload(stem=stem, upload_file=raw.name, image_file=image_file, gray=gray)


def save_heatmap(cam: np.ndarray, stem: str) -> str:
    """Save a Grad-CAM map (float 0..1, same size as the X-ray) as a colored, transparent PNG."""
    c = np.clip(cam, 0, 1)
    color = cv2.applyColorMap((c * 255).astype(np.uint8), cv2.COLORMAP_JET)          # BGR
    alpha = (np.clip((c - 0.15) / 0.85, 0, 1) * 255).astype(np.uint8)                 # weak areas fade out
    name = f"{stem}_heatmap.png"
    _write_png(config.HEATMAP_DIR / name, np.dstack([color, alpha]))
    return name


def overlay(gray: np.ndarray, heatmap_rgba: np.ndarray, strength: float = 0.55) -> np.ndarray:
    """X-ray with the heatmap blended on top (BGR), used for the PDF."""
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    a = heatmap_rgba[..., 3:4].astype(np.float32) / 255 * strength
    return (base * (1 - a) + heatmap_rgba[..., :3].astype(np.float32) * a).astype(np.uint8)


def delete_files(*paths: Path) -> None:
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass
