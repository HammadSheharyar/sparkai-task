"""Evaluation + error analysis on a split (default: test).
Usage: python -m defect.evaluate [--ckpt models/best.pt] [--split test] [--out-dir reports/eval]
Outputs: metrics.json, confusion_matrix.png, pr/roc curves, predictions.csv, threshold_sweep.csv,
         error_analysis.json, errors/*.png (Grad-CAM), error_montage_{fp,fn}.png, RESULTS.md
"""
import argparse
import json
import logging

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from .config import load_config, resolve
from .constants import CLASS_NAMES
from .dataset import CastingDataset
from .gradcam import composite, gradcam
from .logging_config import setup_logging
from .metrics import compute_metrics, plot_confusion, plot_curves
from .model import gradcam_layer, load_checkpoint
from .preprocess import pil_to_array, resize_pil

log = logging.getLogger("evaluate")


def montage(files, out, cols=3, title=""):
    if not files:
        return
    rows = (len(files) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 3.3), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, f in zip(axes.ravel(), files):
        ax.imshow(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB))
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out, dpi=90)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default="models/best.pt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-errors", type=int, default=24)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = resolve(args.out_dir or f"{cfg['paths']['reports_dir']}/eval") / args.split
    (out / "errors").mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=str(out / "evaluate.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(resolve(args.ckpt), device)
    thr, size = ck["threshold"], ck["input_size"]
    csv = resolve(cfg["paths"]["processed_dir"]) / f"{args.split}.csv"
    df = pd.read_csv(csv)
    dl = DataLoader(CastingDataset(csv, size, mean=ck["mean"], std=ck["std"]), batch_size=64, shuffle=False)

    probs = []
    with torch.no_grad():
        for x, _ in dl:
            probs.append(torch.sigmoid(model(x.to(device)).squeeze(1)).cpu().numpy())
    df["prob"] = np.concatenate(probs)
    y = df["label"].values
    df["pred"] = (df["prob"] >= thr).astype(int)
    df["error_type"] = np.select([(df.pred == 1) & (y == 0), (df.pred == 0) & (y == 1)], ["FP", "FN"], "ok")
    df.to_csv(out / "predictions.csv", index=False)

    m_tuned, m_default = compute_metrics(y, df.prob, thr), compute_metrics(y, df.prob, 0.5)
    (out / "metrics.json").write_text(json.dumps({"tuned_threshold": m_tuned, "default_0.5": m_default,
                                                  "report": classification_report(y, df.pred, target_names=CLASS_NAMES, output_dict=True)}, indent=2))
    plot_confusion(m_tuned, out / "confusion_matrix.png", f"{args.split} (tuned)")
    plot_confusion(m_default, out / "confusion_matrix_thr0.5.png", f"{args.split} (default)")
    plot_curves(y, df.prob.values, thr, out)
    sweep = []
    for t in np.linspace(0.05, 0.95, 19):
        m = compute_metrics(y, df.prob, t)
        sweep.append({"threshold": round(float(t), 2), "precision": m["precision"], "recall": m["recall"],
                      "f1": m["f1"], **m["confusion_matrix"]})
    pd.DataFrame(sweep).to_csv(out / "threshold_sweep.csv", index=False)
    log.info("%s tuned: %s", args.split, {k: round(v, 4) for k, v in m_tuned.items() if isinstance(v, float)})

    # ---------------- error analysis ----------------
    layer = gradcam_layer(model, ck["model_name"])
    errs = df[df.error_type != "ok"].copy()
    errs["wrongness"] = np.where(errs.error_type == "FP", errs.prob, 1 - errs.prob)  # most confident mistakes first
    saved = {"FP": [], "FN": []}
    for _, r in errs.sort_values("wrongness", ascending=False).iterrows():
        if len(saved[r.error_type]) >= args.max_errors:
            continue
        with Image.open(r.path) as im:
            img = resize_pil(im, size)
        x = torch.from_numpy(pil_to_array(img, ck["mean"], ck["std"]))[None].to(device)
        cam, p = gradcam(model, layer, x)
        title = f"{r.error_type} true={CLASS_NAMES[int(r.label)]} p_def={p:.3f} {r.path.split('/')[-1][:28]}"
        f = out / "errors" / f"{r.error_type}_{len(saved[r.error_type]):02d}.png"
        cv2.imwrite(str(f), composite(np.asarray(img), cam, title))
        saved[r.error_type].append(f)
    montage(saved["FP"][:9], out / "error_montage_fp.png", title="False positives (normal flagged defective) - most confident first")
    montage(saved["FN"][:9], out / "error_montage_fn.png", title="False negatives (missed defects) - most confident first")

    grp = df.groupby("error_type").agg(n=("prob", "size"), mean_prob=("prob", "mean"),
                                       brightness=("brightness", "mean"), sharpness=("sharpness", "mean")).round(3)
    bins = pd.cut(df["prob"], [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], include_lowest=True)
    calib = df.groupby(bins, observed=True).agg(n=("label", "size"), defective_rate=("label", "mean")).round(3)
    analysis = {"threshold": thr, "n_fp": int((df.error_type == "FP").sum()), "n_fn": int((df.error_type == "FN").sum()),
                "error_vs_correct_stats": grp.to_dict("index"),
                "calibration_by_prob_bin": {str(k): v for k, v in calib.to_dict("index").items()},
                "worst_fp": errs[errs.error_type == "FP"].sort_values("prob", ascending=False).head(10)[["path", "prob"]].to_dict("records"),
                "worst_fn": errs[errs.error_type == "FN"].sort_values("prob").head(10)[["path", "prob"]].to_dict("records")}
    (out / "error_analysis.json").write_text(json.dumps(analysis, indent=2, default=float))

    c = m_tuned["confusion_matrix"]
    (out / "RESULTS.md").write_text(f"""# Results - {args.split} split ({len(df)} images)
Model: `{ck['model_name']}` @ {size}px | operating threshold (chosen on val): **{thr:.4f}**

| metric (defective = positive) | tuned thr | thr 0.5 |
|---|---|---|
| precision | {m_tuned['precision']:.4f} | {m_default['precision']:.4f} |
| recall | {m_tuned['recall']:.4f} | {m_default['recall']:.4f} |
| F1 | {m_tuned['f1']:.4f} | {m_default['f1']:.4f} |
| accuracy | {m_tuned['accuracy']:.4f} | {m_default['accuracy']:.4f} |
| ROC-AUC | {m_tuned['roc_auc']:.4f} | |
| Avg. precision | {m_tuned['average_precision']:.4f} | |

Confusion matrix (tuned): TN={c['tn']} FP={c['fp']} FN={c['fn']} TP={c['tp']}

FN = missed defect (bad part ships). FP = false alarm (good part flagged).
Plots: confusion_matrix.png, pr_curve.png, roc_curve.png | Errors: error_montage_fp.png, error_montage_fn.png, errors/
""")
    log.info("FP=%d FN=%d -> %s", analysis["n_fp"], analysis["n_fn"], out)


if __name__ == "__main__":
    main()
