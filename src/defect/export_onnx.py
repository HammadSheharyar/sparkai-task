"""Export best.pt -> ONNX + metadata JSON, and verify numerical parity vs PyTorch.
Usage: python -m defect.export_onnx [--ckpt models/best.pt] [--out-dir models]
"""
import argparse
import inspect
import json
import logging

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from PIL import Image

from .config import load_config, resolve
from .constants import CLASS_NAMES
from .logging_config import setup_logging
from .model import load_checkpoint
from .preprocess import preprocess

log = logging.getLogger("export")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default="models/best.pt")
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--version", default="1.0.0")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    out = resolve(args.out_dir)
    model, ck = load_checkpoint(resolve(args.ckpt), "cpu")
    size = ck["input_size"]

    kw = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    dummy = torch.randn(1, 3, size, size)
    onnx_path = out / "model.onnx"
    torch.onnx.export(model, dummy, str(onnx_path), input_names=["input"], output_names=["logit"],
                      dynamic_axes={"input": {0: "batch"}, "logit": {0: "batch"}}, opset_version=17, **kw)

    # parity check on real test images
    test = pd.read_csv(resolve(cfg["paths"]["processed_dir"]) / "test.csv").head(16)
    xs = []
    for p in test["path"]:
        with Image.open(p) as im:
            xs.append(preprocess(im, size, ck["mean"], ck["std"]))
    x = np.stack(xs)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy().reshape(-1)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input": x})[0].reshape(-1)
    diff = float(np.abs(ref - got).max())
    log.info("max |logit diff| torch vs onnx = %.2e", diff)
    if diff > 1e-2:
        raise SystemExit(f"ONNX parity check failed ({diff})")

    meta = {"model_name": ck["model_name"], "model_version": args.version, "input_size": size,
            "mean": ck["mean"], "std": ck["std"], "threshold": ck["threshold"], "class_names": CLASS_NAMES,
            "positive_class": "defective", "input_name": "input", "onnx_parity_max_abs_diff": diff,
            "onnx_size_mb": round(onnx_path.stat().st_size / 1e6, 2)}
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2))
    log.info("exported %s (%.1f MB)", onnx_path, meta["onnx_size_mb"])


if __name__ == "__main__":
    main()
