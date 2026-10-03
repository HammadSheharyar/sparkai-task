"""Two-phase transfer learning: (1) head only, (2) full fine-tune with cosine LR + early stopping.
Usage: python -m defect.train [--model efficientnet_b0] [--out-dir models]
"""
import argparse
import json
import logging
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .config import load_config, resolve
from .constants import MEAN, STD
from .dataset import CastingDataset
from .logging_config import setup_logging
from .metrics import compute_metrics, select_threshold
from .model import build_model, load_checkpoint, set_backbone_trainable, split_params

log = logging.getLogger("train")


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=False, desc=""):
    training = optimizer is not None
    model.train(training)
    total, n, probs, ys = 0.0, 0, [], []
    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, desc=desc, leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(x).squeeze(1)
            loss = criterion(logits.float(), y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total += loss.item() * len(y)
            n += len(y)
            probs.append(torch.sigmoid(logits.float()).detach().cpu().numpy())
            ys.append(y.cpu().numpy())
    return total / n, np.concatenate(ys), np.concatenate(probs)


def predict_probs(model, loader, device):
    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in loader:
            out.append(torch.sigmoid(model(x.to(device)).squeeze(1)).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["model"]["name"] = args.model
    if args.no_pretrained:
        cfg["model"]["pretrained"] = False
    out_dir = resolve(args.out_dir or cfg["paths"]["models_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=str(out_dir / "train.log"))
    seed_all(cfg["data"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tc, mc = cfg["train"], cfg["model"]
    amp = bool(tc["amp"]) and device.type == "cuda"
    log.info("device=%s amp=%s model=%s size=%d", device, amp, mc["name"], mc["input_size"])

    proc = resolve(cfg["paths"]["processed_dir"])
    ds_tr = CastingDataset(proc / "train.csv", mc["input_size"], train=True, aug=cfg["augment"])
    ds_va = CastingDataset(proc / "val.csv", mc["input_size"])
    n_pos = int(sum(ds_tr.labels))
    n_neg = len(ds_tr) - n_pos
    log.info("train n=%d (normal=%d defective=%d)", len(ds_tr), n_neg, n_pos)

    imb = tc["imbalance"]
    sampler = None
    if imb in ("sampler", "both"):
        w = np.where(np.array(ds_tr.labels) == 1, 1.0 / n_pos, 1.0 / n_neg)
        sampler = WeightedRandomSampler(w, num_samples=len(ds_tr), replacement=True)
    pos_weight = n_neg / n_pos if imb in ("pos_weight", "both") else 1.0
    log.info("imbalance=%s pos_weight=%.3f sampler=%s", imb, pos_weight, sampler is not None)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    kw = dict(num_workers=tc["num_workers"], pin_memory=device.type == "cuda",
              persistent_workers=tc["num_workers"] > 0)
    dl_tr = DataLoader(ds_tr, tc["batch_size"], sampler=sampler, shuffle=sampler is None,
                       drop_last=True, **kw)
    dl_va = DataLoader(ds_va, tc["batch_size"] * 2, shuffle=False, **kw)

    model = build_model(mc["name"], mc["pretrained"], mc["dropout"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("params=%.2fM", n_params / 1e6)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    backbone, head = split_params(model)

    fe, ne = tc["freeze_epochs"], tc["finetune_epochs"]
    best = {"ap": -1.0, "loss": 1e9, "epoch": 0}
    best_state, bad, hist, opt, sched = None, 0, [], None, None
    t0 = time.time()
    for epoch in range(1, fe + ne + 1):
        if epoch == 1 and fe > 0:
            set_backbone_trainable(model, False)
            opt = torch.optim.AdamW(head, lr=tc["lr_head"], weight_decay=tc["weight_decay"])
            log.info("phase 1: head only")
        if epoch == fe + 1:
            set_backbone_trainable(model, True)
            opt = torch.optim.AdamW([{"params": backbone, "lr": tc["lr_backbone"]},
                                     {"params": head, "lr": tc["lr_head"] * 0.1}],
                                    weight_decay=tc["weight_decay"])
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ne)
            log.info("phase 2: full fine-tune")
        tr_loss, ytr, ptr = run_epoch(model, dl_tr, criterion, device, opt, scaler, amp, f"train {epoch}")
        va_loss, yva, pva = run_epoch(model, dl_va, criterion, device, amp=amp, desc=f"val {epoch}")
        if sched is not None:
            sched.step()
        mv = compute_metrics(yva, pva, 0.5)
        row = {"epoch": epoch, "phase": 1 if epoch <= fe else 2, "train_loss": tr_loss, "val_loss": va_loss,
               "val_ap": mv["average_precision"], "val_auc": mv["roc_auc"], "val_f1@0.5": mv["f1"],
               "val_recall@0.5": mv["recall"], "val_precision@0.5": mv["precision"]}
        hist.append(row)
        log.info(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))

        improved = mv["average_precision"] > best["ap"] + 1e-6 or (
            abs(mv["average_precision"] - best["ap"]) <= 1e-6 and va_loss < best["loss"])
        if improved:
            best = {"ap": mv["average_precision"], "loss": va_loss, "epoch": epoch}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        elif epoch > fe:
            bad += 1
            if bad >= tc["patience"]:
                log.info("early stopping at epoch %d (best epoch %d)", epoch, best["epoch"])
                break

    train_time = time.time() - t0
    model.load_state_dict(best_state)
    pd.DataFrame(hist).to_csv(out_dir / "history.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    h = pd.DataFrame(hist)
    ax[0].plot(h.epoch, h.train_loss, label="train")
    ax[0].plot(h.epoch, h.val_loss, label="val")
    ax[0].set_title("loss")
    ax[0].legend()
    ax[1].plot(h.epoch, h.val_ap, label="val AP")
    ax[1].plot(h.epoch, h["val_f1@0.5"], label="val F1@0.5")
    ax[1].set_title("validation")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=120)
    plt.close()

    # operating threshold chosen on VALIDATION only (never on test)
    pva = predict_probs(model, dl_va, device)
    thr = select_threshold(np.array(ds_va.labels), pva, cfg["threshold"]["min_recall"])
    val_metrics = compute_metrics(ds_va.labels, pva, thr)
    log.info("best epoch=%d val_ap=%.4f | threshold=%.4f -> val %s", best["epoch"], best["ap"], thr,
             {k: round(v, 4) for k, v in val_metrics.items() if isinstance(v, float)})

    ckpt = {"model_name": mc["name"], "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "input_size": mc["input_size"], "mean": list(MEAN), "std": list(STD), "threshold": thr,
            "best_epoch": best["epoch"], "config": cfg}
    torch.save(ckpt, out_dir / "best.pt")
    (out_dir / "train_summary.json").write_text(json.dumps({
        "model": mc["name"], "params_millions": round(n_params / 1e6, 3), "train_seconds": round(train_time, 1),
        "best_epoch": best["epoch"], "pos_weight": pos_weight, "imbalance": imb,
        "threshold": thr, "val_metrics_at_threshold": val_metrics}, indent=2))
    load_checkpoint(out_dir / "best.pt")  # sanity: checkpoint reloads
    log.info("saved %s", out_dir / "best.pt")


if __name__ == "__main__":
    main()
