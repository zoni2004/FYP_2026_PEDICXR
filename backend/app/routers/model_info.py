from fastapi import APIRouter, Request

from .. import config
from ..schemas import MetricOut, ModelInfoOut
from ..services.model_info import load_results

router = APIRouter(prefix="/api", tags=["model"])


@router.get("/model-info", response_model=ModelInfoOut, summary="Current model and its test results")
def model_info(request: Request):
    predictor = getattr(request.app.state, "predictor", None)
    summary, folds = load_results()
    return ModelInfoOut(
        name=getattr(predictor, "name", config.MODEL_NAME),
        checkpoint=config.MODEL_PATH.name,
        loaded=predictor is not None,
        uses_age_sex=bool(getattr(predictor, "uses_age_sex", False)),
        threshold=config.THRESHOLD,
        findings=list(getattr(predictor, "findings", ["Pneumonia"])),
        test_metrics={k: MetricOut(**v) for k, v in summary.items()},
        folds=folds,
        notes="ResNet-50 trained on PediCXR (pneumonia vs normal), 5-fold cross-validation, "
              "tested on the official PediCXR test split (170 pneumonia, 907 normal).",
    )
