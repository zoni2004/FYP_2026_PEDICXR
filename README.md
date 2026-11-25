# FYP_2026_PEDICXR

Age-aware AI system for pediatric chest X-rays (FYP 2026).

| Folder | What it is |
|---|---|
| `cxr_cap/` | Model training code and the trained ResNet-50 weights (`fold0_best.pt`) |
| `data/` | PediCXR label, annotation and age/sex CSV files |
| `backend/` | FastAPI server: upload an X-ray, run the model, Grad-CAM heatmap, draft report, PDF, SQLite database |
| `frontend/` | React web app: worklist, upload, study viewer, report approval, model info |

## Run the web app

Start the backend, then the frontend

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8010
```

```powershell
npm.cmd run dev
```
