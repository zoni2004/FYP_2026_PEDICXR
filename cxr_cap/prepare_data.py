
import argparse
from pathlib import Path

import cv2
import numpy as np
import pydicom
from pydicom.pixels import apply_voi_lut
from tqdm import tqdm


def dicom_to_uint8(path: Path) -> np.ndarray:
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    try:
        arr = apply_voi_lut(arr, ds).astype(np.float32)
    except Exception:
        pass
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    lo, hi = np.percentile(arr, (0.5, 99.5))
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    return (arr * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--max_side", type=int, default=1024)
    args = ap.parse_args()

    root, out = Path(args.data_root), Path(args.out_root)
    for split in ["train", "test"]:
        src = root / split
        dst = out / split
        dst.mkdir(parents=True, exist_ok=True)
        files = [p for p in src.iterdir() if p.suffix.lower() in (".dicom", ".dcm")]
        print(f"[{split}] {len(files)} DICOM files mile")
        bad = 0
        for p in tqdm(files):
            target = dst / f"{p.stem}.png"
            if target.exists():
                continue
            try:
                img = dicom_to_uint8(p)
                h, w = img.shape
                s = args.max_side / max(h, w)
                if s < 1:
                    img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(target), img)
            except Exception as e:
                bad += 1
                print("FAIL", p.name, e)
        print(f"[{split}] done, failed={bad}")


if __name__ == "__main__":
    main()
