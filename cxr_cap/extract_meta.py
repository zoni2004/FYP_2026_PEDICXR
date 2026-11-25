"""Read age + sex from the original DICOM headers -> meta.csv (image_id, age_months, sex).
Usage:
  python extract_meta.py --dicom_root "D:\\FYP\\PediCXR dataset" --out ../data/meta.csv
"""
import argparse
import re
from pathlib import Path

import pandas as pd
import pydicom
from tqdm import tqdm


def parse_age(s):
    """DICOM age string ('010M', '001Y', '003W', '012D') -> months (float) or None."""
    m = re.match(r"^\s*(\d+)\s*([DWMY])\s*$", str(s).upper())
    if not m:
        return None
    n, u = int(m.group(1)), m.group(2)
    return {"D": n / 30.44, "W": n * 7 / 30.44, "M": float(n), "Y": n * 12.0}[u]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom_root", required=True)
    ap.add_argument("--out", default="meta.csv")
    a = ap.parse_args()

    files = [p for p in Path(a.dicom_root).rglob("*") if p.suffix.lower() in (".dicom", ".dcm")]
    print("DICOM files found:", len(files))
    rows = []
    for p in tqdm(files):
        d = pydicom.dcmread(p, stop_before_pixels=True)
        rows.append({"image_id": p.stem, "age_months": parse_age(d.get("PatientAge")),
                     "sex": str(d.get("PatientSex") or "").upper()[:1]})
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print("saved", a.out)
    print("missing age:", int(df.age_months.isna().sum()), "| sex counts:", df.sex.value_counts().to_dict())
    print(df.age_months.describe().round(1))


if __name__ == "__main__":
    main()