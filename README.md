# FYP_2026_PEDICXR

Age-aware AI system for pediatric chest X-rays (FYP 2026).

| Folder | What it is |
|---|---|
| `cxr_cap/` | Model training code and the trained ResNet-50 weights (`fold0_best.pt`) |
| `data/` | PediCXR label, annotation and age/sex CSV files |
| `backend/` | FastAPI server: upload an X-ray, run the model, Grad-CAM heatmap, draft report, PDF, SQLite database |
| `frontend/` | React web app: worklist, upload, study viewer, report approval, model info |

## Run the web app

Start the backend, then the frontend, each in its own terminal (Windows PowerShell):

```powershell
cd backend; .venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8010
```

```powershell
cd frontend; npm.cmd run dev
```

Open http://localhost:5173. First-time setup is in [backend/README.md](backend/README.md)
and [frontend/README.md](frontend/README.md).
