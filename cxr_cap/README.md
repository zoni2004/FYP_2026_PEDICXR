# Pediatric CAP — paper implementation on PediCXR (VS Code)

Paper: Wang et al., "Multimodal-based auxiliary diagnosis for pediatric community acquired pneumonia", Sci Rep 2025.

## Setup (VS Code terminal, PowerShell)
    python -m venv venv
    venv\Scripts\activate
    # GPU torch: https://pytorch.org/get-started/locally  (CUDA wala command)
    pip install -r requirements.txt
    python smoke_test.py          # dataset ke baghair shapes/augmentations check

## Run
    python prepare_data.py --data_root "D:/FYP/PediCXR dataset/pedi-cxr" --out_root "D:/FYP/pedi_png"
    # quick test
    python train.py --data_root "D:/FYP/PediCXR dataset/pedi-cxr" --img_root "D:/FYP/pedi_png" --epochs 2 --folds_to_run 0 --limit 300
    # full
    python train.py --data_root "D:/FYP/PediCXR dataset/pedi-cxr" --img_root "D:/FYP/pedi_png" --epochs 100
    # heatmap
    python gradcam.py --ckpt runs/exp1/fold0_best.pt --image "D:/FYP/pedi_png/test/<id>.png"

Outputs in runs/exp1: fold_results.csv, summary_mean_std.csv, roc_oof.png, foldK_best.pt
