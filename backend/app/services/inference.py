"""Runs the trained chest X-ray model and makes the Grad-CAM heatmap.

The model code comes from cxr_cap/ (model.py, dataset.py). To use a new model later,
write another class with the same `predict()` and set it in main.py.
"""
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .. import config
from .storage import add_cxr_cap_to_path


@dataclass
class Prediction:
    findings: dict[str, float]   # finding name -> probability
    cam: np.ndarray              # float 0..1, same height/width as the input X-ray
    peak: tuple[float, float]    # where the heatmap is strongest, as (x, y) fractions of width/height


class Predictor(Protocol):
    name: str
    findings: list[str]
    uses_age_sex: bool

    def predict(self, gray: np.ndarray, age_months: int, sex: str) -> Prediction: ...


class CXRPredictor:
    """ResNet-50 pneumonia classifier from cxr_cap (pneumonia vs normal)."""

    findings = ["Pneumonia"]

    def __init__(self, checkpoint: Path, name: str = config.MODEL_NAME, size: int = config.IMG_SIZE):
        add_cxr_cap_to_path()
        import torch
        from model import MultimodalCAP

        self._torch = torch
        self.name, self.size = name, size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(checkpoint, map_location=self.device)
        self.uses_age_sex = any(k.startswith("meta.") for k in state)   # age/sex-aware checkpoint?
        self.model = MultimodalCAP(use_cxr=True, use_meta=self.uses_age_sex, pretrained=False).to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self._lock = threading.Lock()   # Grad-CAM hooks are not safe to run twice at once

    def predict(self, gray: np.ndarray, age_months: int, sex: str) -> Prediction:
        import cv2
        import torch.nn.functional as F
        from dataset import encode_meta, letterbox_eval, to_tensor

        torch = self._torch
        ih, iw = gray.shape
        img = letterbox_eval(np.stack([gray] * 3, axis=-1), self.size)
        # where the X-ray sits inside the padded square (same maths as letterbox_eval)
        s = min(self.size / iw, self.size / ih)
        nw, nh = int(iw * s), int(ih * s)
        dx, dy = (self.size - nw) // 2, (self.size - nh) // 2

        x = to_tensor(img).unsqueeze(0).to(self.device)
        meta = None
        if self.uses_age_sex:
            meta = torch.from_numpy(encode_meta(age_months, sex)).unsqueeze(0).to(self.device)

        acts, grads = {}, {}
        layer = self.model.cam_layer
        with self._lock:
            h1 = layer.register_forward_hook(lambda m, i, o: acts.update(v=o))
            h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.update(v=go[0]))
            try:
                with torch.enable_grad():
                    logits = self.model(cxr=x, meta=meta)
                    prob = F.softmax(logits.float(), 1)[0, 1].item()
                    self.model.zero_grad(set_to_none=True)
                    logits[0, 1].backward()
            finally:
                h1.remove()
                h2.remove()
                self.model.zero_grad(set_to_none=True)

        w = grads["v"].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * acts["v"]).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(self.size, self.size), mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()[dy:dy + nh, dx:dx + nw]          # drop the padding
        cam = cv2.resize(cam, (iw, ih), interpolation=cv2.INTER_LINEAR)   # back to X-ray size
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        py, px = np.unravel_index(int(np.argmax(cam)), cam.shape)
        return Prediction(findings={"Pneumonia": float(prob)}, cam=cam.astype(np.float32),
                          peak=(px / iw, py / ih))
