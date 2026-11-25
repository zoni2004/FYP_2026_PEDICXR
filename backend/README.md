# PediCXR backend (FastAPI)

The API behind the web app. It takes a chest X-ray with the child's age and sex,
runs the trained model from `../cxr_cap`, makes a Grad-CAM heatmap and a draft
report, and saves everything in a SQLite database.

## Run it

From this `backend/` folder (Windows PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
$env:DISABLE_SQLALCHEMY_CEXT=1
python -m pip install sqlalchemy==2.0.36 --no-binary sqlalchemy
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8010
```

Open http://localhost:8010/docs. Every endpoint can be tried there, including
uploading an X-ray.

The model file `../cxr_cap/fold0_best.pt` must exist (it comes with the repo).

### Notes for Windows 11 with Smart App Control

Smart App Control blocks program files that Windows doesn't recognise yet. On this
laptop it blocked: the newest PyTorch (2.15), SQLAlchemy 2.1's compiled module,
the `pylibjpeg-libjpeg` plugin and the `fastapi.exe` / `uvicorn.exe` launchers.
The steps above avoid all of them:

- PyTorch 2.6.0 loads fine.
- SQLAlchemy 2.0 is installed as pure Python (`--no-binary`).
- Start the server with `python -m uvicorn` instead of `fastapi dev`.
- Don't install `pylibjpeg-libjpeg` (only needed for JPEG-compressed DICOM files).

Port 8000 is used by another program on this laptop, so the backend runs on **8010**.
The React app should call `http://localhost:8010`.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET  | `/api/health` | Server is up, model is loaded |
| POST | `/api/studies` | Upload X-ray (`image`, `age_months`, `sex`, optional `patient_ref`), analyze it |
| GET  | `/api/studies?status=review&q=...` | Worklist: to-review first, highest pneumonia probability first |
| GET  | `/api/studies/{id}` | One study with findings, heatmap link and report |
| PUT  | `/api/studies/{id}/report` | Save the report (`status: draft`) or approve it (`status: approved`) |
| POST | `/api/studies/{id}/feedback` | Doctor agrees / disagrees with a finding |
| GET  | `/api/studies/{id}/pdf` | Download the report as a PDF |
| GET  | `/api/model-info` | Current model and its saved test results |
| GET  | `/files/uploads/...`, `/files/heatmaps/...` | X-ray and heatmap images |

## Folders

```
app/
  main.py              app setup, CORS, file serving, loads the model at start
  config.py            settings (all can be changed with PEDICXR_* environment variables)
  db.py, models.py     SQLite database and tables
  schemas.py           request / response shapes
  routers/             the endpoints
  services/
    inference.py       runs the model + Grad-CAM
    storage.py         saves uploads (PNG, JPG, DICOM) and heatmaps
    report.py          draft report text (template for now)
    pdf.py             PDF report
    model_info.py      reads cxr_cap/results/*.csv
storage/               created on first run: uploads/, heatmaps/, db/  (not in git)
tests/                 pytest tests (use a fake model, no weights needed)
```

## Tests

```powershell
python -m pytest
```

## Settings

| Variable | Default |
|---|---|
| `PEDICXR_MODEL` | `../cxr_cap/fold0_best.pt` |
| `PEDICXR_MODEL_NAME` | `resnet50-fold0` |
| `PEDICXR_THRESHOLD` | `0.5` |
| `PEDICXR_STORAGE` | `backend/storage` |
| `PEDICXR_CORS` | `http://localhost:5173,http://127.0.0.1:5173` (the React dev server) |
| `PEDICXR_MAX_UPLOAD_MB` | `25` |

## Swapping in a new model later

Write a class in `services/inference.py` with the same `predict(gray, age_months, sex)`
that returns a `Prediction` (findings, heatmap, peak), and load it in `main.py`.
For the GPT-2 / T5 report, replace `draft_report()` in `services/report.py`.
The API and the frontend do not change.

Not in FYP-1: login, roles, encryption and audit logs (FYP-2).
