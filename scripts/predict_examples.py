"""Generate example predictions from the held-out test split.
Usage: python scripts/predict_examples.py [--n 5] [--url http://localhost:8000]   (no --url => local ONNX)
Writes examples/predictions.json, examples/predictions.md and copies the images to examples/images/.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from defect.inference import Predictor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="images per class")
    ap.add_argument("--url", default=None)
    ap.add_argument("--api-key", default=None)
    a = ap.parse_args()
    test = pd.read_csv(ROOT / "data/processed/test.csv")
    pick = pd.concat([test[test.label == k].sample(min(a.n, (test.label == k).sum()), random_state=1) for k in (0, 1)])
    img_dir = ROOT / "examples/images"
    img_dir.mkdir(parents=True, exist_ok=True)
    predictor = None if a.url else Predictor(ROOT / "models/model.onnx", ROOT / "models/model_meta.json")
    rows = []
    for _, r in pick.iterrows():
        p = Path(r.path)
        data = p.read_bytes()
        if predictor:
            out = predictor.predict_bytes(data)
        else:
            import requests
            h = {"X-API-Key": a.api_key} if a.api_key else {}
            resp = requests.post(f"{a.url}/predict", files={"file": (p.name, data)}, headers=h, timeout=30)
            resp.raise_for_status()
            out = resp.json()
        shutil.copy(p, img_dir / p.name)
        rows.append({"image": f"images/{p.name}", "true_class": "defective" if r.label else "normal", **out})
    (ROOT / "examples/predictions.json").write_text(json.dumps(rows, indent=2))
    md = ["| image | true | predicted | confidence | p(defect) | total ms |", "|---|---|---|---|---|---|"]
    for x in rows:
        ok = "OK" if x["true_class"] == x["predicted_class"] else "WRONG"
        md.append(f"| ![]({x['image']}) | {x['true_class']} | {x['predicted_class']} ({ok}) | {x['confidence']} | "
                  f"{x['defect_probability']} | {x['latency']['total_ms']} |")
    (ROOT / "examples/predictions.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
