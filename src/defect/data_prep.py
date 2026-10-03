"""Dataset analysis (EDA), cleaning, leakage-safe stratified split.

Usage: python -m defect.data_prep [--config configs/config.yaml]

Steps
 1. Scan raw_dir recursively; label from parent folder (def_front/ok_front) or file name (cast_def_* / cast_ok_*).
 2. Per-image stats (size, brightness, sharpness, md5, 256-bit dHash); unreadable files are dropped.
 3. Exact-duplicate removal (md5) + near-duplicate grouping (dHash hamming) -> groups never cross splits.
 4. Stratified GROUP split into train/val/test, CSVs written to processed_dir.
 5. EDA plots + summary.json in reports/eda.
"""
import argparse
import hashlib
import json
import logging
import re
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import load_config, resolve
from .constants import CLASS_NAMES, IMG_EXTS
from .logging_config import setup_logging

log = logging.getLogger("data_prep")


def infer_label(p: Path):
    """1 = defective, 0 = normal, None = unknown. Parent folder first, then file stem."""
    for c in (p.parent.name.lower(), p.stem.lower()):
        if "def" in c:
            return 1
        if re.search(r"(^|[_\-\s])ok($|[_\-\s])", c):
            return 0
    return None


def dhash256(gray: np.ndarray) -> np.ndarray:
    small = cv2.resize(gray, (17, 16), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()  # 256 bools


def scan(raw_dir: Path):
    rows, skipped, corrupt = [], [], []
    for p in sorted(raw_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        label = infer_label(p)
        if label is None:
            skipped.append(str(p))
            continue
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.size == 0:
            corrupt.append(str(p))
            continue
        rel = p.relative_to(raw_dir).parts
        rows.append({
            "path": str(p.resolve()), "label": label,
            "orig_split": next((x for x in rel if x.lower() in ("train", "test")), ""),
            "md5": hashlib.md5(p.read_bytes()).hexdigest(),
            "height": gray.shape[0], "width": gray.shape[1],
            "brightness": float(gray.mean()),
            "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "_hash": dhash256(gray),
        })
    return rows, skipped, corrupt


def group_near_duplicates(hashes: np.ndarray, thr: int) -> np.ndarray:
    n = len(hashes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n - 1):
        d = (hashes[i + 1:] != hashes[i]).sum(1)
        for j in np.nonzero(d <= thr)[0]:
            ri, rj = find(i), find(i + 1 + int(j))
            if ri != rj:
                parent[rj] = ri
    return np.array([find(i) for i in range(n)])


def eda_plots(df: pd.DataFrame, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    counts = df["label"].value_counts().sort_index()
    plt.figure(figsize=(4, 3))
    plt.bar([CLASS_NAMES[i] for i in counts.index], counts.values, color=["#4c9", "#e55"])
    for i, v in enumerate(counts.values):
        plt.text(i, v, str(v), ha="center", va="bottom")
    plt.title("Class distribution (after dedup)")
    plt.tight_layout()
    plt.savefig(out / "class_distribution.png", dpi=120)
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    for k, col in enumerate(["brightness", "sharpness"]):
        for lab in (0, 1):
            ax[k].hist(df.loc[df.label == lab, col], bins=40, alpha=0.6, label=CLASS_NAMES[lab])
        ax[k].set_title(col)
        ax[k].legend()
    plt.tight_layout()
    plt.savefig(out / "brightness_sharpness.png", dpi=120)
    plt.close()

    fig, axes = plt.subplots(2, 8, figsize=(20, 5.2))
    for r, lab in enumerate((0, 1)):
        sub = df[df.label == lab].sample(min(8, (df.label == lab).sum()), random_state=0)
        for c in range(8):
            axes[r, c].axis("off")
            if c < len(sub):
                axes[r, c].imshow(cv2.cvtColor(cv2.imread(sub.iloc[c]["path"]), cv2.COLOR_BGR2RGB))
                axes[r, c].set_title(CLASS_NAMES[lab], fontsize=9)
    plt.tight_layout()
    plt.savefig(out / "samples.png", dpi=100)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    raw_dir, proc_dir = resolve(cfg["paths"]["raw_dir"]), resolve(cfg["paths"]["processed_dir"])
    eda_dir = resolve(cfg["paths"]["reports_dir"]) / "eda"
    setup_logging()
    proc_dir.mkdir(parents=True, exist_ok=True)
    eda_dir.mkdir(parents=True, exist_ok=True)

    rows, skipped, corrupt = scan(raw_dir)
    if not rows:
        raise SystemExit(f"No labelled images found under {raw_dir}. See README (Dataset).")
    hashes = np.stack([r.pop("_hash") for r in rows])
    df = pd.DataFrame(rows)
    n_raw = len(df)
    log.info("scanned %d images (%d unlabeled skipped, %d corrupt)", n_raw, len(skipped), len(corrupt))

    # label conflicts: identical bytes with different labels
    conflicts = df.groupby("md5")["label"].nunique()
    conflict_md5 = set(conflicts[conflicts > 1].index)
    if conflict_md5:
        log.warning("%d md5 hashes appear with conflicting labels -> dropped", len(conflict_md5))
        keep = ~df["md5"].isin(conflict_md5)
        df, hashes = df[keep].reset_index(drop=True), hashes[keep.values]

    dup_mask = df.duplicated("md5", keep="first")
    n_exact = int(dup_mask.sum())
    df, hashes = df[~dup_mask].reset_index(drop=True), hashes[~dup_mask.values]
    log.info("removed %d exact duplicates", n_exact)

    df["group"] = group_near_duplicates(hashes, cfg["data"]["near_dup_hamming"])
    gs = df["group"].value_counts()
    log.info("near-dup groups: %d groups, largest=%d, groups with >1 image=%d",
             len(gs), gs.max(), int((gs > 1).sum()))
    if gs.max() > 0.02 * len(df):
        log.warning("largest near-dup group is >2%% of data; lower data.near_dup_hamming if images are merely similar")
    # label-pure groups only (a mixed group would be ambiguous)
    mixed = df.groupby("group")["label"].nunique()
    if (mixed > 1).any():
        log.warning("%d near-dup groups have mixed labels (kept; visually similar normal/defective pairs)",
                    int((mixed > 1).sum()))

    seed = cfg["data"]["seed"]
    y, g = df["label"].values, df["group"].values
    idx = np.arange(len(df))
    f1 = StratifiedGroupKFold(cfg["data"]["n_test_splits"], shuffle=True, random_state=seed)
    rest, test = next(f1.split(idx, y, g))
    f2 = StratifiedGroupKFold(cfg["data"]["n_val_splits"], shuffle=True, random_state=seed)
    tr, va = next(f2.split(rest, y[rest], g[rest]))
    splits = {"train": rest[tr], "val": rest[va], "test": test}
    for a in splits:
        for b in splits:
            if a < b:
                assert not set(g[splits[a]]) & set(g[splits[b]]), f"group leakage {a}/{b}"

    summary = {
        "n_raw_labeled": n_raw, "unlabeled_skipped": len(skipped), "corrupt": len(corrupt),
        "exact_duplicates_removed": n_exact, "label_conflicts_dropped": len(conflict_md5),
        "n_final": len(df),
        "class_counts": {CLASS_NAMES[k]: int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "imbalance_ratio_majority_to_minority": round(float(df.label.value_counts().max() / df.label.value_counts().min()), 3),
        "image_sizes": {f"{w}x{h}": int(c) for (w, h), c in df.groupby(["width", "height"]).size().items()},
        "brightness_by_class": df.groupby("label")["brightness"].describe().round(2).to_dict("index"),
        "sharpness_by_class": df.groupby("label")["sharpness"].describe().round(2).to_dict("index"),
        "near_dup_groups": {"n_groups": int(len(gs)), "largest": int(gs.max()),
                            "groups_gt1": int((gs > 1).sum())},
        "splits": {},
    }
    for name, ix in splits.items():
        part = df.iloc[ix].sample(frac=1, random_state=seed)
        part.to_csv(proc_dir / f"{name}.csv", index=False)
        vc = part["label"].value_counts().sort_index()
        summary["splits"][name] = {"n": len(part), **{CLASS_NAMES[k]: int(v) for k, v in vc.items()},
                                   "defective_frac": round(float(part.label.mean()), 3)}
        log.info("%-5s n=%d %s", name, len(part), summary["splits"][name])

    eda_plots(df, eda_dir)
    (eda_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    if skipped:
        (eda_dir / "skipped_unlabeled.txt").write_text("\n".join(skipped))
    log.info("done -> %s, %s", proc_dir, eda_dir)


if __name__ == "__main__":
    main()
