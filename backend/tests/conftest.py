import os
import tempfile

# Use a throw-away storage folder and database for tests (must be set before the app is imported).
os.environ["PEDICXR_STORAGE"] = tempfile.mkdtemp(prefix="pedicxr-test-")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.services.inference import Prediction  # noqa: E402


class FakePredictor:
    """Stands in for the real model so tests run fast and without the weights file."""
    name = "fake-model"
    findings = ["Pneumonia"]
    uses_age_sex = False

    def __init__(self):
        self.next_probability = 0.8

    def predict(self, gray, age_months, sex):
        h, w = gray.shape
        yy, xx = np.mgrid[0:h, 0:w]
        cam = np.exp(-(((xx - w * 0.3) / (w * 0.15)) ** 2 + ((yy - h * 0.6) / (h * 0.15)) ** 2)).astype(np.float32)
        return Prediction(findings={"Pneumonia": self.next_probability}, cam=cam, peak=(0.3, 0.6))


@pytest.fixture(scope="session")
def fake():
    return FakePredictor()


@pytest.fixture(scope="session")
def client(fake):
    app.state.predictor = fake
    with TestClient(app) as c:
        yield c


def xray_png(w=300, h=360) -> bytes:
    img = np.zeros((h, w), np.uint8)
    cv2.ellipse(img, (w // 2, h // 2), (w // 3, h // 3), 0, 0, 360, 160, -1)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()
