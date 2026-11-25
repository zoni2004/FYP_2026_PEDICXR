"""PediCXR backend. Run from the backend/ folder:

    fastapi dev app/main.py          (or: uvicorn app.main:app --reload)

Then open http://localhost:8000/docs to try every endpoint.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import config
from .db import init_db
from .routers import model_info, studies

log = logging.getLogger("pedicxr")
config.ensure_dirs()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if getattr(app.state, "predictor", None) is None:   # tests put a fake model here first
        try:
            from .services.inference import CXRPredictor
            app.state.predictor = CXRPredictor(config.MODEL_PATH)
            log.info("Model loaded: %s (age/sex: %s)", app.state.predictor.name, app.state.predictor.uses_age_sex)
        except Exception:
            log.exception("Could not load the model from %s", config.MODEL_PATH)
            app.state.predictor = None
    yield


app = FastAPI(title="PediCXR API", version="0.1.0", lifespan=lifespan,
              description="Backend for the pediatric chest X-ray web app (FYP-1).")
app.state.predictor = None

app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_credentials=False,
                   allow_methods=["GET", "POST", "PUT"], allow_headers=["*"],
                   expose_headers=["Content-Disposition"])

# Only these two folders are served; the database folder is not.
app.mount("/files/uploads", StaticFiles(directory=config.UPLOAD_DIR), name="uploads")
app.mount("/files/heatmaps", StaticFiles(directory=config.HEATMAP_DIR), name="heatmaps")

app.include_router(studies.router)
app.include_router(model_info.router)


@app.get("/api/health", tags=["health"], summary="Is the server up and the model loaded?")
def health():
    p = app.state.predictor
    return {"status": "ok", "model_loaded": p is not None, "model": getattr(p, "name", None)}
