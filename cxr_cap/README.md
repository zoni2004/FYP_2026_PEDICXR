# Pediatric Pneumonia Detection from Chest X-rays (CXR-only)

Implementation of the chest X-ray branch of the paper
"Multimodal-based auxiliary diagnosis for pediatric community acquired
pneumonia" (Wang et al., Scientific Reports, 2025), trained on the public
PediCXR dataset.

## What this project does
The paper combines three inputs: chest X-ray, lab tests and clinical text.
PediCXR only has X-ray images and labels, so we trained only the X-ray part
(modified ResNet-50). The lab-test (DNN) and clinical-text (Bi-GRU) branches
and the fusion classifier are also written in `model.py`, but they are NOT
trained, because that data is not public. So this is the paper's X-ray-only
baseline, not the full multimodal model.

## Task
Binary classification: pneumonia vs normal.
- Pneumonia = any of: Pneumonia, Brocho-pneumonia, Pleuro-pneumonia
- Normal = "No finding". Other diseases are excluded (as in the paper).
- The training set is undersampled to 1:1. The test set is NOT balanced
  (170 pneumonia, 907 normal, about 84% normal), so look at AUC and recall,
  not only accuracy.

## What follows the paper
- Augmentations: random scale and padding, rotation (25%, +-10 deg),
  HSV color jitter
- ResNet-50 (ImageNet weights) + FC layer to 128 features
- AdamW, lr 1e-4, cross-entropy
- 5-fold cross-validation, best epoch chosen by lowest validation loss

## Differences from the paper
- Different dataset (PediCXR) and X-ray only. Results are not comparable
  to the paper's multimodal 94.2% accuracy.
- Fewer epochs (20 for fold 0, 10 for folds 1-4) instead of 100. The best
  epoch was between 4 and 7 in every fold.
- Image-level split (PediCXR has no patient ID).
- Some sizes the paper does not give (GRU hidden size, embedding size)
  are our own assumptions.
- The paper's text says the fused vector has 178 features, but
  128 + 32 + 32 = 192. We used 192.

## Results (5-fold cross-validation, mean ± std over folds)
Test set = official PediCXR test split (not balanced). Each fold's best
model (lowest validation loss) is evaluated on it.

| AUC | Recall | Specificity | Precision | F1 | Accuracy |
|-----|--------|-------------|-----------|----|----------|
| 0.820 ± 0.020 | 0.706 ± 0.076 | 0.768 ± 0.072 | 0.373 ± 0.056 | 0.483 ± 0.042 | 0.758 ± 0.052 |

Validation (balanced 1:1, closer to the paper's setting): AUC 0.823 ± 0.017,
accuracy 0.733 ± 0.017, precision 0.770 ± 0.047, recall 0.678 ± 0.108,
F1 0.714 ± 0.042. These are slightly optimistic, because the best epoch is
picked on the same validation set.

For reference, the paper reports for its X-ray-only ResNet-50 (own private,
balanced dataset): accuracy 0.846, precision 0.837, recall 0.863, F1 0.849,
AUC 0.908. Our numbers are lower, but the datasets are different, so this is
not a direct comparison.

Precision on the test set is low mainly because about 84% of the test images
are normal, so even a small false-positive rate gives many false positives.

Per-fold numbers: `results/fold_results_all5.csv`.
Summary: `results/summary_5fold.csv`.
ROC curves: `results/roc_fold0.png` (fold 0) and
`results/roc_folds1to4.png` (folds 1-4). Each covers only its own folds.

## Grad-CAM examples
`results/cams/` has 8 Grad-CAM examples from the fold 0 model: the 4
pneumonia and 4 normal images with the most confident correct predictions,
picked from a random sample of 120 test images. They are not random picks,
so they look better than the typical case. In some examples the model
highlights the lungs, but in others it highlights the shoulders or abdomen,
so it may partly use non-lung cues. We did not check this systematically.

## Files
- `prepare_data.py` - converts DICOM to PNG
- `dataset.py` - labels and augmentations
- `model.py` - model (CXR, lab and text branches + fusion)
- `train.py` - k-fold training and evaluation
- `gradcam.py` - Grad-CAM heatmaps
- `smoke_test.py` - quick shape check, no dataset needed
- `results/` - result tables, ROC curves, example heatmaps

## Setup
Training was run on Google Colab with a T4 GPU.

    
    Google Collab training link
    https://colab.research.google.com/drive/1wGaUQmzw5rxPvgAGjnbSgVlM5gzJKiZ3
    python -m venv venv
    venv\Scripts\activate
    # Install PyTorch with CUDA from https://pytorch.org/get-started/locally
    pip install -r requirements.txt
    python smoke_test.py

