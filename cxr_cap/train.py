"""
5-fold cross-validation training (paper setup):
  AdamW(lr=1e-4, betas=(0.9,0.999)), cross-entropy, 100 epochs, 512x512 input,
  har fold ka best epoch = LOWEST validation loss, final result = 5 folds ka average.
Extra: har fold ka best model official PediCXR test split par bhi evaluate hota hai.

Quick test:   python train.py --data_root ".../pedi-cxr" --img_root ".../pedi_png" --epochs 2 --folds_to_run 0 --limit 300
Full run:     python train.py --data_root ".../pedi-cxr" --img_root ".../pedi_png" --epochs 100
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve, confusion_matrix)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from dataset import PediCXRDataset, build_binary_labels
from model import MultimodalCAP


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"acc": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "specificity": tn / max(tn + fp, 1),
            "auc": roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")}


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    ys, ps, loss_sum, n = [], [], 0.0, 0
    for b in loader:
        x, y = b["cxr"].to(device), b["label"].to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(cxr=x)
        loss_sum += criterion(logits.float(), y).item() * len(y)
        n += len(y)
        ys.append(y.cpu().numpy())
        ps.append(F.softmax(logits.float(), 1)[:, 1].cpu().numpy())
    return loss_sum / n, np.concatenate(ys), np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="pedi-cxr folder (CSV files yahan hain)")
    ap.add_argument("--img_root", default=None, help="prepare_data.py ka output; None = data_root (DICOM direct)")
    ap.add_argument("--out_dir", default="runs/exp1")
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--folds_to_run", type=int, nargs="*", default=None, help="e.g. 0 1 (time bachane ke liye)")
    ap.add_argument("--neg_mode", default="no_finding", choices=["no_finding", "all_other"])
    ap.add_argument("--balance_ratio", type=float, default=1.0,
                    help="negatives = ratio * positives (undersample). 0 = off. Paper dataset balanced thi (797 vs 793)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="debug: itni images hi use karo")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    root = Path(args.data_root)
    img_root = Path(args.img_root) if args.img_root else root
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tr_df, pos_cols = build_binary_labels(root / "image_labels_train.csv", neg_mode=args.neg_mode)
    te_df, _ = build_binary_labels(root / "image_labels_test.csv", neg_mode=args.neg_mode)
    print("positive columns used:", pos_cols)
    print("train  :", tr_df.label.value_counts().to_dict(), "| test:", te_df.label.value_counts().to_dict())

    if args.balance_ratio > 0:
        pos, neg = tr_df[tr_df.label == 1], tr_df[tr_df.label == 0]
        n_neg = min(len(neg), int(len(pos) * args.balance_ratio))
        tr_df = pd.concat([pos, neg.sample(n_neg, random_state=args.seed)]).sample(frac=1, random_state=args.seed)
        print("after undersampling:", tr_df.label.value_counts().to_dict())
    if args.limit:
        tr_df = tr_df.groupby("label", group_keys=False).apply(
            lambda d: d.sample(min(len(d), args.limit // 2), random_state=args.seed)).reset_index(drop=True)
        te_df = te_df.sample(min(len(te_df), args.limit // 2), random_state=args.seed)
    tr_df = tr_df.reset_index(drop=True)

    test_ds = PediCXRDataset(te_df, img_root / "test", args.img_size, train=False)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=args.workers)
    # PediCXR mein har image alag study hai (patient id CSV mein nahi) -> image-level split.

    skf = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    fold_res, oof_y, oof_p = [], [], []
    for k, (tr_idx, va_idx) in enumerate(skf.split(tr_df, tr_df.label)):
        if args.folds_to_run is not None and k not in args.folds_to_run:
            continue
        print(f"\n===== FOLD {k} =====")
        tr_ds = PediCXRDataset(tr_df.iloc[tr_idx], img_root / "train", args.img_size, train=True)
        va_ds = PediCXRDataset(tr_df.iloc[va_idx], img_root / "train", args.img_size, train=False)
        tr_loader = DataLoader(tr_ds, args.batch_size, shuffle=True, num_workers=args.workers,
                               drop_last=True, pin_memory=device.type == "cuda")
        va_loader = DataLoader(va_ds, args.batch_size, shuffle=False, num_workers=args.workers)

        model = MultimodalCAP(use_cxr=True, use_lab=False, use_text=False).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
        scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
        crit = torch.nn.CrossEntropyLoss()

        best = {"val_loss": 1e9}
        for ep in range(1, args.epochs + 1):
            model.train()
            t0, run = time.time(), 0.0
            for b in tr_loader:
                x, y = b["cxr"].to(device, non_blocking=True), b["label"].to(device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(cxr=x)
                loss = crit(logits.float(), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                run += loss.item()
            vl, vy, vp = evaluate(model, va_loader, device, crit)
            m = metrics(vy, vp)
            print(f"ep {ep:3d} | train {run / len(tr_loader):.4f} | val {vl:.4f} | acc {m['acc']:.3f} "
                  f"rec {m['recall']:.3f} auc {m['auc']:.3f} | {time.time() - t0:.0f}s")
            if vl < best["val_loss"]:                     # paper: lowest validation loss
                best = {"val_loss": vl, "epoch": ep, **m, "y": vy, "p": vp}
                torch.save(model.state_dict(), out / f"fold{k}_best.pt")

        model.load_state_dict(torch.load(out / f"fold{k}_best.pt", map_location=device))
        _, ty, tp = evaluate(model, test_loader, device, crit)
        tm = metrics(ty, tp)
        print(f"fold {k} best ep {best['epoch']} | val acc {best['acc']:.3f} | TEST acc {tm['acc']:.3f} auc {tm['auc']:.3f}")
        fold_res.append({"fold": k, "best_epoch": best["epoch"], "val_loss": best["val_loss"],
                         **{f"val_{a}": best[a] for a in ("acc", "precision", "recall", "f1", "specificity", "auc")},
                         **{f"test_{a}": v for a, v in tm.items()}})
        oof_y.append(best["y"])
        oof_p.append(best["p"])

    res = pd.DataFrame(fold_res)
    res.to_csv(out / "fold_results.csv", index=False)
    summary = res.drop(columns=["fold"]).agg(["mean", "std"]).T
    summary.to_csv(out / "summary_mean_std.csv")
    print("\n===== SUMMARY (mean +- std over folds) =====\n", summary.round(4))

    if oof_y:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        y, p = np.concatenate(oof_y), np.concatenate(oof_p)
        fpr, tpr, _ = roc_curve(y, p)
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f"CXR (ResNet-50) AUC={roc_auc_score(y, p):.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("1 - Specificity")
        plt.ylabel("Sensitivity")
        plt.legend()
        plt.savefig(out / "roc_oof.png", dpi=150, bbox_inches="tight")
    json.dump(vars(args), open(out / "args.json", "w"), indent=2)


if __name__ == "__main__":
    main()
