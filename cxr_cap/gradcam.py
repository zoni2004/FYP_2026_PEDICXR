"""
GradCAM (paper Fig. 9): trained fold checkpoint par CAP class ka heatmap.
    python gradcam.py --ckpt runs/exp1/fold0_best.pt --image "D:/.../test/xxxx.png" --out cam.png
"""
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dataset import letterbox_eval, to_tensor
from model import MultimodalCAP


def load_gray(path):
    if str(path).lower().endswith((".dicom", ".dcm")):
        from prepare_data import dicom_to_uint8
        g = dicom_to_uint8(__import__("pathlib").Path(path))
    else:
        g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return np.stack([g, g, g], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="gradcam.png")
    ap.add_argument("--size", type=int, default=512)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalCAP(use_cxr=True, pretrained=False).to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev))
    model.eval()

    acts, grads = {}, {}
    layer = model.cam_layer
    layer.register_forward_hook(lambda m, i, o: acts.update(v=o))
    layer.register_full_backward_hook(lambda m, gi, go: grads.update(v=go[0]))

    img = letterbox_eval(load_gray(a.image), a.size)
    x = to_tensor(img).unsqueeze(0).to(dev)
    logits = model(cxr=x)
    prob = F.softmax(logits, 1)[0, 1].item()
    model.zero_grad()
    logits[0, 1].backward()                                    # target = CAP class

    w = grads["v"].mean(dim=(2, 3), keepdim=True)              # channel weights
    cam = F.relu((w * acts["v"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=img.shape[:2], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    heat = cv2.applyColorMap((cam.detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_JET)
    over = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0.55, heat, 0.45, 0)
    cv2.putText(over, f"P(CAP)={prob:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(a.out, np.concatenate([cv2.cvtColor(img, cv2.COLOR_RGB2BGR), over], axis=1))
    print("saved", a.out, "P(CAP) =", round(prob, 4))


if __name__ == "__main__":
    main()
