import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)

from .constants import CLASS_NAMES


def compute_metrics(y, p, thr: float) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    div = lambda a, b: float(a / b) if b else 0.0
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    both = len(set(y.tolist())) > 1
    return {
        "threshold": float(thr), "n": int(len(y)),
        "accuracy": div(tp + tn, len(y)), "precision": prec, "recall": rec,
        "f1": div(2 * prec * rec, prec + rec), "specificity": div(tn, tn + fp),
        "false_positive_rate": div(fp, fp + tn), "false_negative_rate": div(fn, fn + tp),
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "average_precision": float(average_precision_score(y, p)) if both else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def select_threshold(y, p, min_recall=0.98) -> float:
    """Largest threshold whose recall on 'defective' >= min_recall (max precision at that recall).
    Falls back to the F1-optimal threshold if the recall target is unreachable."""
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.nonzero(rec[:-1] >= min_recall)[0]
    if len(ok):
        return float(thr[ok].max())
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    return float(thr[int(np.argmax(f1))])


def plot_confusion(m: dict, path, title="Confusion matrix"):
    c = m["confusion_matrix"]
    cm = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n{cm[i, j] / max(cm[i].sum(), 1):.1%}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], [f"pred {n}" for n in CLASS_NAMES])
    ax.set_yticks([0, 1], [f"true {n}" for n in CLASS_NAMES])
    ax.set_title(f"{title} (thr={m['threshold']:.3f})", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_curves(y, p, thr, out_dir):
    prec, rec, _ = precision_recall_curve(y, p)
    plt.figure(figsize=(4.5, 3.8))
    plt.plot(rec, prec)
    plt.xlabel("recall (defective)")
    plt.ylabel("precision (defective)")
    plt.title("Precision-Recall")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curve.png", dpi=130)
    plt.close()
    fpr, tpr, _ = roc_curve(y, p)
    plt.figure(figsize=(4.5, 3.8))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=130)
    plt.close()
