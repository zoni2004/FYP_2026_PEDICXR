"""Streamlit demo UI for the CXR-only pediatric pneumonia model.
Run:  streamlit run app.py
"""
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F

from dataset import letterbox_eval, to_tensor, encode_meta
from model import MultimodalCAP

HERE = Path(__file__).parent
SIZE = 512
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="Pediatric Pneumonia Detection", page_icon="🫁", layout="wide")


# ---------------------------------------------------------------- helpers
def find_checkpoints():
    pats = ["*.pt", "runs/*.pt", "runs/*/*.pt", "../runs/*.pt", "../*.pt", "weights/*.pt"]
    found = []
    for p in pats:
        found += list(HERE.glob(p))
    return sorted(set(found))


@st.cache_resource(show_spinner="Loading model...")
def load_model(path):
    sd = torch.load(path, map_location=DEV)
    use_meta = any(k.startswith("meta.") for k in sd)      # age/sex-aware checkpoint?
    m = MultimodalCAP(use_cxr=True, use_meta=use_meta, pretrained=False).to(DEV)
    m.load_state_dict(sd)
    return m.eval()


def read_image(uploaded):
    data = uploaded.getvalue()
    if uploaded.name.lower().endswith((".dcm", ".dicom")):
        from prepare_data import dicom_to_uint8
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as f:
            f.write(data)
        g = dicom_to_uint8(Path(f.name))
    else:
        g = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise ValueError("Could not read this image.")
    return np.stack([g, g, g], -1)


def predict_with_cam(model, rgb, meta=None):
    img = letterbox_eval(rgb, SIZE)
    x = to_tensor(img).unsqueeze(0).to(DEV)
    acts, grads = {}, {}
    layer = model.cam_layer
    h1 = layer.register_forward_hook(lambda m, i, o: acts.update(v=o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.update(v=go[0]))
    try:
        with torch.enable_grad():
            logits = model(cxr=x, meta=meta)
            prob = F.softmax(logits, 1)[0, 1].item()
            model.zero_grad()
            logits[0, 1].backward()
    finally:
        h1.remove()
        h2.remove()
    w = grads["v"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * acts["v"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=img.shape[:2], mode="bilinear", align_corners=False)[0, 0]
    cam = ((cam - cam.min()) / (cam.max() - cam.min() + 1e-8)).detach().cpu().numpy()
    heat = cv2.cvtColor(cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET),
                        cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img, 0.55, heat, 0.45, 0)
    return img, overlay, prob


# ---------------------------------------------------------------- sidebar
st.sidebar.title("⚙️ Settings")
ckpts = find_checkpoints()
ckpt = None
if ckpts:
    ckpt = st.sidebar.selectbox("Model checkpoint", ckpts, format_func=lambda p: p.name)
else:
    st.sidebar.warning("No .pt checkpoint found. Upload one below.")
up_ckpt = st.sidebar.file_uploader("…or upload checkpoint (.pt)", type=["pt"])
if up_ckpt:
    tmp = Path(tempfile.gettempdir()) / up_ckpt.name
    tmp.write_bytes(up_ckpt.getvalue())
    ckpt = tmp
thr = st.sidebar.slider("Decision threshold (P(pneumonia))", 0.05, 0.95, 0.50, 0.05)
st.sidebar.caption(f"Device: **{DEV.type.upper()}**")
st.sidebar.info("Research prototype (X-ray only baseline). Not for clinical use.")

# ---------------------------------------------------------------- header
st.title("🫁 Pediatric Pneumonia Detection from Chest X-rays")
st.caption("Base paper: Wang et al., *Scientific Reports* 2025 · ResNet-50 CXR branch · trained on PediCXR")

tab1, tab2, tab3 = st.tabs(["🔍 Diagnose", "📊 Model results", "ℹ️ About"])

# ---------------------------------------------------------------- tab 1
with tab1:
    f = st.file_uploader("Upload a chest X-ray (PNG / JPG / DICOM)",
                         type=["png", "jpg", "jpeg", "dcm", "dicom"])
    ca, cb, cc = st.columns(3)
    yrs = ca.number_input("Age: years", 0, 18, 1)
    mos = cb.number_input("Age: months", 0, 11, 0)
    sex = cc.selectbox("Sex", ["F", "M"])
    if f is None:
        st.info("Upload an X-ray to get a prediction and a Grad-CAM heatmap.")
    elif ckpt is None:
        st.error("No model checkpoint available. Put `fold0_best.pt` in `cxr_cap/` or upload it in the sidebar.")
    else:
        try:
            model = load_model(str(ckpt))
            rgb = read_image(f)
            meta = None
            if model.use_meta:
                meta = torch.from_numpy(encode_meta(yrs * 12 + mos, sex)).unsqueeze(0).to(DEV)
            else:
                st.caption("This checkpoint is image-only: age and sex are ignored.")
            with st.spinner("Analyzing..."):
                img, overlay, prob = predict_with_cam(model, rgb, meta)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

        is_pn = prob >= thr
        c1, c2, c3 = st.columns([1, 1, 1])
        c1.image(img, caption="Input (preprocessed)", use_container_width=True)
        c2.image(overlay, caption="Grad-CAM (target: pneumonia)", use_container_width=True)
        with c3:
            if is_pn:
                st.error("### Pneumonia detected")
            else:
                st.success("### Normal")
            st.metric("P(pneumonia)", f"{prob:.1%}")
            st.progress(float(prob))
            st.caption(f"Threshold: {thr:.2f}")
            st.warning("Heatmaps may highlight non-lung areas (shoulders / abdomen) — "
                       "treat as indicative only.")

# ---------------------------------------------------------------- tab 2
with tab2:
    res = HERE / "results"
    st.subheader("5-fold cross-validation (official PediCXR test split)")
    try:
        s = pd.read_csv(res / "summary_5fold.csv", index_col=0)
        keys = ["test_auc", "test_recall", "test_specificity", "test_precision", "test_f1", "test_acc"]
        cols = st.columns(len(keys))
        for c, k in zip(cols, keys):
            c.metric(k.replace("test_", "").upper(), f"{s.loc[k, 'mean']:.3f}", f"± {s.loc[k, 'std']:.3f}",
                     delta_color="off")
        with st.expander("Full summary table"):
            st.dataframe(s.round(4))
    except Exception:
        st.info("results/summary_5fold.csv not found.")
    st.caption("Test set is ~84% normal, so precision is low; focus on AUC and recall.")

    st.subheader("ROC curves")
    a, b = st.columns(2)
    for col, name in ((a, "roc_fold0.png"), (b, "roc_folds1to4.png")):
        if (res / name).exists():
            col.image(str(res / name), use_container_width=True)

    st.subheader("Per-fold results")
    if (res / "fold_results_all5.csv").exists():
        st.dataframe(pd.read_csv(res / "fold_results_all5.csv").round(4), use_container_width=True)

    cams = sorted((res / "cams").glob("*.png")) if (res / "cams").exists() else []
    if cams:
        st.subheader("Grad-CAM examples (fold 0)")
        cols = st.columns(2)
        for i, p in enumerate(cams):
            cols[i % 2].image(str(p), caption=p.stem, use_container_width=True)

# ---------------------------------------------------------------- tab 3
with tab3:
    st.markdown("""
**Task:** binary classification, pneumonia vs normal (PediCXR).
**Model:** ResNet-50 (ImageNet weights) → FC 128 → classifier. AdamW, lr 1e-4, cross-entropy.
**Training:** 5-fold CV, best epoch by lowest validation loss, undersampled 1:1.

**Limitations**
- X-ray only (the paper's lab-test and clinical-text branches need private data).
- Different dataset from the paper, so results are not directly comparable.
- Image-level split (no patient IDs in PediCXR).
- Research prototype, not a medical device.
""")
