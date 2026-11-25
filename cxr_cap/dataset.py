
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ----------------------------------------------------------------- age / sex meta
MAX_AGE_MONTHS = 216.0   # 18 years


def encode_meta(age_months, sex):
    """-> np.float32 [age_norm, sex, age_missing]. age_norm = log1p(months)/log1p(216); sex M=1, F=0, unknown=0.5."""
    miss = 0.0
    if age_months is None or (isinstance(age_months, float) and np.isnan(age_months)):
        age_months, miss = 12.0, 1.0
    age_norm = float(np.log1p(min(max(age_months, 0.0), MAX_AGE_MONTHS)) / np.log1p(MAX_AGE_MONTHS))
    sx = {"M": 1.0, "F": 0.0}.get(str(sex).strip().upper()[:1], 0.5)
    return np.array([age_norm, sx, miss], dtype=np.float32)


def load_meta(csv_path):
    """meta.csv (image_id, age_months, sex) -> {image_id: encoded vector}"""
    df = pd.read_csv(csv_path)
    return {str(r.image_id): encode_meta(r.age_months, r.sex) for r in df.itertuples()}


# ----------------------------------------------------------------- labels
def build_binary_labels(csv_path, pos_keyword="pneumonia", neg_mode="no_finding"):

    df = pd.read_csv(csv_path)
    id_col = next(c for c in df.columns if c.lower().replace(" ", "_") in ("image_id", "imageid", "id"))
    cols = [c for c in df.columns if c != id_col]
    pos_cols = [c for c in cols if pos_keyword in c.lower()]
    if not pos_cols:
        raise ValueError(f"'{pos_keyword}' wala koi column nahi mila. Columns: {cols}")
    # kuch versions mein 0/1, kuch mein True/False
    num = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    is_pos = (num[pos_cols].sum(axis=1) > 0)

    if neg_mode == "no_finding":
        nf_cols = [c for c in cols if "no finding" in c.lower() or "normal" in c.lower()]
        if not nf_cols:
            raise ValueError(f"'No finding' column nahi mila. Columns: {cols}  (--neg_mode all_other try karo)")
        is_neg = (num[nf_cols].sum(axis=1) > 0) & (~is_pos)
        keep = is_pos | is_neg
    else:
        keep = pd.Series(True, index=df.index)

    out = pd.DataFrame({"image_id": df.loc[keep, id_col].astype(str), "label": is_pos[keep].astype(int)})
    return out.reset_index(drop=True), pos_cols


def resolve_path(img_dir: Path, image_id: str):
    for ext in (".png", ".dicom", ".dcm", ".jpg"):
        p = img_dir / f"{image_id}{ext}"
        if p.exists():
            return p
    return None


# ----------------------------------------------------------------- augmentations
def rand(a=0.0, b=1.0):
    return random.uniform(a, b)


def letterbox_eval(img, size):
    """Eval: aspect ratio preserve karke resize + gray padding (random kuch nahi)."""
    ih, iw = img.shape[:2]
    s = min(size / iw, size / ih)
    nw, nh = int(iw * s), int(ih * s)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((size, size, 3), 128, dtype=np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = img
    return canvas


def random_scale_pad(img, size, jitter=0.3, scale_range=(0.8, 1.2)):
    """
    Paper Eq (1)-(3). w,h = NETWORK INPUT size (512), original image nahi
    (ye YOLO-style augmentation hai jo paper ne use ki).
    """
    ih, iw = img.shape[:2]
    w = h = size
    ar_new = (iw / ih) * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)   # Eq (1)
    scale = rand(*scale_range)                                                         # Eq (2)
    if ar_new < 1:                                                                     # Eq (3)
        nh = int(scale * h)
        nw = int(nh * ar_new)
    else:
        nw = int(scale * w)
        nh = int(nw / ar_new)
    nw, nh = max(nw, 8), max(nh, 8)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((h, w, 3), 128, dtype=np.uint8)
    dx = int(rand(0, w - nw)) if w >= nw else int(rand(w - nw, 0))
    dy = int(rand(0, h - nh)) if h >= nh else int(rand(h - nh, 0))
    # paste with cropping agar image canvas se badi ho
    x0, y0 = max(dx, 0), max(dy, 0)
    x1, y1 = min(dx + nw, w), min(dy + nh, h)
    canvas[y0:y1, x0:x1] = img[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return canvas


def random_rotate(img, p=0.25, max_deg=10):
    if random.random() > p:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rand(-max_deg, max_deg), 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128, 128, 128))


def hsv_augment(img, h=0.1, s=0.7, v=0.3):
    """Eq (4): r = u * max_offset + 1, u ~ U(-1,1); H,S,V ko multiply karte hain."""
    rh, rs, rv = [rand(-1, 1) * m + 1 for m in (h, s, v)]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] * rh) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rs, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * rv, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def to_tensor(img):
    x = (img.astype(np.float32) / 255.0 - IMNET_MEAN) / IMNET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


# ----------------------------------------------------------------- dataset
class PediCXRDataset(Dataset):
    """
    __getitem__ dict return karta hai taake lab/text modality baad mein plug ho sake:
        {"cxr": (3,512,512), "lab": (8,) [optional], "text": (27,) [optional], "label": int}
    PediCXR mein lab/text hota hi nahi, isliye default sirf cxr + label.
    """

    def __init__(self, df, img_dir, size=512, train=False, aug_scale=True, meta=None):
        self.df, self.size, self.train = df.reset_index(drop=True), size, train
        self.meta = meta   # dict image_id -> vector, or None (image-only)
        self.img_dir = Path(img_dir)
        self.paths = [resolve_path(self.img_dir, i) for i in self.df["image_id"]]
        missing = sum(p is None for p in self.paths)
        if missing:
            print(f"WARNING: {missing} images folder mein nahi mili ({self.img_dir}); unhein drop kar raha hoon")
            keep = [i for i, p in enumerate(self.paths) if p is not None]
            self.df = self.df.iloc[keep].reset_index(drop=True)
            self.paths = [self.paths[i] for i in keep]
        self.aug_scale = aug_scale

    def __len__(self):
        return len(self.df)

    def _load(self, p):
        if p.suffix.lower() in (".dicom", ".dcm"):
            from prepare_data import dicom_to_uint8
            g = dicom_to_uint8(p)
        else:
            g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        return np.stack([g, g, g], axis=-1)  # 3-channel (ImageNet pretrained ResNet ke liye)

    def __getitem__(self, i):
        img = self._load(self.paths[i])
        if self.train:
            img = random_scale_pad(img, self.size) if self.aug_scale else letterbox_eval(img, self.size)
            img = random_rotate(img)
            img = hsv_augment(img)
        else:
            img = letterbox_eval(img, self.size)
        out = {"cxr": to_tensor(img), "label": int(self.df.loc[i, "label"])}
        if self.meta is not None:
            vec = self.meta.get(str(self.df.loc[i, "image_id"]), encode_meta(None, None))
            out["meta"] = torch.from_numpy(vec)
        return out
