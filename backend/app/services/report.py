"""Draft report text.

FYP-1: a fixed template filled from the model's result.
Later: replace `draft_report` with the GPT-2 / T5 report writer; the API stays the same.
"""


def format_age(months: int) -> str:
    if months < 24:
        return f"{months} month{'s' if months != 1 else ''}"
    y, m = divmod(months, 12)
    return f"{y} years" + (f" {m} months" if m else "")


def _location(peak: tuple[float, float]) -> str:
    x, y = peak
    side = "right" if x < 0.5 else "left"   # on a frontal X-ray the patient's right is on the image's left
    zone = "upper" if y < 0.4 else "mid" if y < 0.62 else "lower"
    return f"{side} {zone} zone"


def draft_report(age_months: int, sex: str, probability: float, threshold: float,
                 peak: tuple[float, float]) -> str:
    patient = f"{format_age(age_months)}, {'male' if sex == 'M' else 'female'}"
    pct = f"{probability * 100:.0f}%"
    head = f"EXAMINATION: Chest X-ray, frontal view.\nPATIENT: {patient}.\n\n"
    if probability >= threshold:
        where = _location(peak)
        return head + (
            "FINDINGS:\n"
            f"Airspace opacity in the {where}, in keeping with pneumonia.\n"
            "Other findings (effusion, pneumothorax, heart size) are not assessed by this model.\n\n"
            "IMPRESSION:\n"
            f"Findings suggest pneumonia in the {where} (AI probability {pct}).\n"
            "Clinical correlation advised."
        )
    return head + (
        "FINDINGS:\n"
        "No focal consolidation detected by the model.\n"
        "Other findings (effusion, pneumothorax, heart size) are not assessed by this model.\n\n"
        "IMPRESSION:\n"
        f"No radiographic evidence of pneumonia (AI probability {pct})."
    )
