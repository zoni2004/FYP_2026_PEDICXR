
import numpy as np
import torch

from dataset import random_scale_pad, random_rotate, hsv_augment, letterbox_eval, to_tensor
from model import MultimodalCAP

img = (np.random.rand(1200, 900, 3) * 255).astype(np.uint8)
for _ in range(50):
    a = hsv_augment(random_rotate(random_scale_pad(img, 512)))
    assert a.shape == (512, 512, 3) and a.dtype == np.uint8
assert letterbox_eval(img, 512).shape == (512, 512, 3)
print("augmentations OK")

m = MultimodalCAP(use_cxr=True, use_lab=True, use_text=True, pretrained=False).eval()
out = m(cxr=torch.randn(2, 3, 512, 512), lab=torch.randn(2, 8),
        text=torch.randint(0, 643, (2, 27)))
assert out.shape == (2, 2)
print("full multimodal (CXR+Lab+Text) forward OK, fused dim =", m.classifier[0].in_features)
m2 = MultimodalCAP(use_cxr=True, pretrained=False).eval()
assert m2(cxr=torch.randn(2, 3, 512, 512)).shape == (2, 2)
print("CXR-only forward OK")
