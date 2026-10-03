"""Latency / size benchmark: PyTorch vs ONNX Runtime (CPU, + CUDA if present), batch 1 and 8.
Usage: python -m defect.benchmark
"""
import argparse
import json
import time

import numpy as np
import onnxruntime as ort
import torch

from .config import resolve
from .model import load_checkpoint


def timeit(fn, n=100, warm=10):
    for _ in range(warm):
        fn()
    t = []
    for _ in range(n):
        s = time.perf_counter()
        fn()
        t.append((time.perf_counter() - s) * 1000)
    t = np.array(t)
    return {"p50_ms": round(float(np.percentile(t, 50)), 2), "p95_ms": round(float(np.percentile(t, 95)), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/best.pt")
    ap.add_argument("--onnx", default="models/model.onnx")
    ap.add_argument("--out", default="reports/benchmark.json")
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    model, ck = load_checkpoint(resolve(a.ckpt), "cpu")
    s = ck["input_size"]
    res = {"input_size": s, "params_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
           "onnx_size_mb": round(resolve(a.onnx).stat().st_size / 1e6, 2), "runs": {}}
    for b in (1, 8):
        x = np.random.randn(b, 3, s, s).astype(np.float32)
        xt = torch.from_numpy(x)
        with torch.no_grad():
            r = timeit(lambda: model(xt), a.n)
        res["runs"][f"pytorch_cpu_b{b}"] = r | {"img_per_s": round(b * 1000 / r["p50_ms"], 1)}
        for prov in [p for p in ("CPUExecutionProvider", "CUDAExecutionProvider") if p in ort.get_available_providers()]:
            sess = ort.InferenceSession(str(resolve(a.onnx)), providers=[prov])
            r = timeit(lambda: sess.run(None, {"input": x}), a.n)
            res["runs"][f"onnx_{prov[:4].lower()}_b{b}"] = r | {"img_per_s": round(b * 1000 / r["p50_ms"], 1)}
    out = resolve(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
