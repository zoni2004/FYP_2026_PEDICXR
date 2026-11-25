"""/api/studies: upload an X-ray, list and open studies, edit/approve reports, feedback, PDF."""
from typing import Annotated, Literal, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import config
from ..db import get_db
from ..models import Feedback, Report, Result, Study
from ..schemas import FeedbackIn, FeedbackOut, FindingOut, ReportIn, ReportOut, StudyOut, StudySummary
from ..services import storage
from ..services.pdf import build_pdf
from ..services.report import draft_report

router = APIRouter(prefix="/api/studies", tags=["studies"])


def get_predictor(request: Request):
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(503, "The AI model is not loaded. Check the server log.")
    return predictor


def _pneumonia(study: Study) -> Optional[float]:
    for r in reversed(study.results):
        if r.finding == "Pneumonia":
            return r.probability
    return None


def to_summary(study: Study) -> StudySummary:
    return StudySummary(id=study.id, code=study.code, patient_ref=study.patient_ref, age_months=study.age_months,
                        sex=study.sex, status=study.status, created_at=study.created_at,
                        pneumonia_probability=_pneumonia(study))


def to_out(study: Study) -> StudyOut:
    latest_model = study.results[-1].model_name if study.results else None
    latest = [r for r in study.results if r.model_name == latest_model]
    heat = next((r.heatmap_file for r in latest if r.heatmap_file), None)
    feedback = {}
    for fb in study.feedback:          # later answers replace earlier ones
        feedback[fb.finding] = fb.agree
    return StudyOut(
        **to_summary(study).model_dump(),
        original_name=study.original_name,
        image_url=f"/files/uploads/{study.image_file}",
        heatmap_url=f"/files/heatmaps/{heat}" if heat else None,
        model_name=latest_model,
        threshold=config.THRESHOLD,
        findings=[FindingOut(name=r.finding, probability=r.probability, positive=r.probability >= config.THRESHOLD)
                  for r in latest],
        report=ReportOut(text=study.report.text, status=study.report.status, updated_at=study.report.updated_at)
        if study.report else None,
        feedback=feedback,
    )


def _get_study(db: Session, study_id: int) -> Study:
    study = db.get(Study, study_id)
    if study is None:
        raise HTTPException(404, f"Study {study_id} was not found.")
    return study


@router.post("", status_code=201, response_model=StudyOut, summary="Upload an X-ray and analyze it")
async def create_study(
    image: Annotated[UploadFile, File(description="Chest X-ray: PNG, JPG or DICOM")],
    age_months: Annotated[int, Form(ge=0, le=216, description="Age in months (0 to 216 = 18 years)")],
    sex: Annotated[Literal["M", "F"], Form()],
    patient_ref: Annotated[str, Form(max_length=64, description="Hospital number, not a name")] = "",
    db: Session = Depends(get_db),
    predictor=Depends(get_predictor),
):
    data = await image.read(config.MAX_UPLOAD_MB * 1024 * 1024 + 1)
    try:
        saved = await run_in_threadpool(storage.save_upload, image.filename or "", data)
    except storage.UploadError as e:
        raise HTTPException(e.status_code, str(e)) from e

    files = [config.UPLOAD_DIR / saved.upload_file, config.UPLOAD_DIR / saved.image_file]
    try:
        pred = await run_in_threadpool(predictor.predict, saved.gray, age_months, sex)
        heatmap_file = await run_in_threadpool(storage.save_heatmap, pred.cam, saved.stem)
        files.append(config.HEATMAP_DIR / heatmap_file)

        study = Study(patient_ref=patient_ref.strip() or None, age_months=age_months, sex=sex,
                      original_name=(image.filename or "upload")[:255], upload_file=saved.upload_file,
                      image_file=saved.image_file, status="review")
        for name, prob in pred.findings.items():
            study.results.append(Result(model_name=predictor.name, finding=name, probability=prob,
                                        heatmap_file=heatmap_file))
        p = pred.findings.get("Pneumonia", 0.0)
        study.report = Report(text=draft_report(age_months, sex, p, config.THRESHOLD, pred.peak), status="draft")
        db.add(study)
        db.commit()
        db.refresh(study)
    except Exception as e:
        db.rollback()
        storage.delete_files(*files)
        raise HTTPException(500, "The X-ray could not be analyzed. Check the server log for details.") from e
    return to_out(study)


@router.get("", response_model=list[StudySummary], summary="List studies for the worklist")
def list_studies(
    status: Optional[Literal["review", "approved"]] = None,
    q: Annotated[Optional[str], Query(max_length=64, description="Search study code or patient ref")] = None,
    db: Session = Depends(get_db),
):
    stmt = select(Study)
    if status:
        stmt = stmt.where(Study.status == status)
    studies = db.scalars(stmt).all()
    if q:
        needle = q.strip().lower()
        studies = [s for s in studies if needle in s.code.lower() or needle in (s.patient_ref or "").lower()]
    # to-review first, then highest pneumonia probability, then newest
    studies.sort(key=lambda s: (s.status != "review", -(_pneumonia(s) or 0.0), -s.id))
    return [to_summary(s) for s in studies]


@router.get("/{study_id}", response_model=StudyOut, summary="Open one study")
def get_study(study_id: int, db: Session = Depends(get_db)):
    return to_out(_get_study(db, study_id))


@router.put("/{study_id}/report", response_model=StudyOut, summary="Save or approve the report")
def update_report(study_id: int, body: ReportIn, db: Session = Depends(get_db)):
    study = _get_study(db, study_id)
    if study.report and study.report.status == "approved":
        raise HTTPException(409, "This report is already approved and can't be changed.")
    if study.report is None:
        study.report = Report(text=body.text, status=body.status)
    else:
        study.report.text = body.text
        study.report.status = body.status
    if body.status == "approved":
        study.status = "approved"
    db.commit()
    db.refresh(study)
    return to_out(study)


@router.post("/{study_id}/feedback", response_model=FeedbackOut, status_code=201,
             summary="Doctor agrees or disagrees with a finding")
def add_feedback(study_id: int, body: FeedbackIn, db: Session = Depends(get_db)):
    study = _get_study(db, study_id)
    names = {r.finding for r in study.results}
    if body.finding not in names:
        raise HTTPException(400, f"'{body.finding}' is not a finding of this study. Use one of: {', '.join(sorted(names))}.")
    fb = Feedback(finding=body.finding, agree=body.agree)
    study.feedback.append(fb)
    db.commit()
    db.refresh(fb)
    return FeedbackOut(finding=fb.finding, agree=fb.agree, created_at=fb.created_at)


@router.get("/{study_id}/pdf", summary="Download the report as a PDF",
            responses={200: {"content": {"application/pdf": {}}}})
def study_pdf(study_id: int, db: Session = Depends(get_db)):
    study = _get_study(db, study_id)
    pdf = build_pdf(study)
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{study.code}-report.pdf"'})
