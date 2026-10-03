from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
