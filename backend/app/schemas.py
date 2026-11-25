"""Request and response shapes. This is the contract the React app relies on."""
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class FindingOut(BaseModel):
    name: str
    probability: float
    positive: bool                      # probability >= threshold


class ReportOut(BaseModel):
    text: str
    status: Literal["draft", "approved"]
    updated_at: datetime


class StudySummary(BaseModel):
    id: int
    code: str                           # e.g. PCX-0007
    patient_ref: Optional[str]
    age_months: int
    sex: Literal["M", "F"]
    status: Literal["review", "approved"]
    created_at: datetime
    pneumonia_probability: Optional[float]


class StudyOut(StudySummary):
    original_name: str
    image_url: str                      # relative to the API address, e.g. /files/uploads/x.png
    heatmap_url: Optional[str]
    model_name: Optional[str]
    threshold: float
    findings: list[FindingOut]
    report: Optional[ReportOut]
    feedback: dict[str, bool]           # finding -> doctor agrees?
    uncertainty: None = None            # filled in FYP-2


class ReportIn(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    status: Literal["draft", "approved"] = "draft"


class FeedbackIn(BaseModel):
    finding: str = Field(min_length=1, max_length=64)
    agree: bool


class FeedbackOut(BaseModel):
    finding: str
    agree: bool
    created_at: datetime


class MetricOut(BaseModel):
    mean: float
    std: float


class ModelInfoOut(BaseModel):
    name: str
    checkpoint: str
    loaded: bool
    uses_age_sex: bool
    threshold: float
    findings: list[str]
    test_metrics: dict[str, MetricOut]   # auc, recall, specificity, precision, f1, acc
    folds: list[dict[str, float]]
    notes: str
