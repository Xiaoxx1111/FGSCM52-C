from copy import deepcopy
from pathlib import Path
import yaml


def merge(a, b):
    out = deepcopy(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else deepcopy(v)
    return out


def load_config(path):
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text())
    base = data.pop("extends", None)
    cfg = merge(load_config(path.parent / base), data) if base else data
    # Dataset/output paths are relative to the directory from which commands run.
    return cfg
