from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

MEAN = (.48145466, .4578275, .40821073)
STD = (.26862954, .26130258, .27577711)
CONDITIONS = ["fog", "gaussian_noise", "impulse_noise", "motion_blur", "defocus_blur",
              "low_resolution", "jpeg", "cloud", "sea_clutter", "stripe", "sar_style", "thermal_ir_style"]
HOLDOUTS = {"optical_quality": CONDITIONS[:7], "environmental": CONDITIONS[7:10],
            "noise": CONDITIONS[1:3], "blur": CONDITIONS[3:5],
            "resolution_jpeg": CONDITIONS[5:7], "sensor_style": CONDITIONS[10:]}


@dataclass(frozen=True)
class Record:
    source_id: str
    split: str
    class_id: int
    condition: str
    severity: int
    path: str


def read_metadata(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if not d.get("groups") or not d.get("classes"):
        raise ValueError("classes.json requires nonempty groups and classes")
    for i, c in enumerate(d["classes"]):
        if c.get("id") != i or not c.get("name") or not 0 <= c.get("group_id", -1) < len(d["groups"]):
            raise ValueError("Class IDs must be contiguous from zero with valid names and group IDs")
    if len(set(x["name"] for x in d["classes"])) != len(d["classes"]):
        raise ValueError("Duplicate class names")
    if len(set(d["groups"])) != len(d["groups"]):
        raise ValueError("Duplicate coarse-group names")
    return d


def read_manifest(path, metadata, check_files=False, root=None):
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"source_id", "split", "class_id", "condition", "severity", "path"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Manifest is missing columns: {required-set(reader.fieldnames or [])}")
        rows = [Record(r["source_id"], r["split"], int(r["class_id"]), r["condition"], int(r["severity"]), r["path"]) for r in reader]
    validate_records(rows, metadata, check_files, root)
    return rows


def validate_records(rows, metadata, check_files=False, root=None):
    sources, keys, clean, paths = {}, set(), set(), {}
    for r in rows:
        if not r.source_id or r.split not in {"train", "val", "test"}:
            raise ValueError("Every record requires source_id and train/val/test split")
        if not 0 <= r.class_id < len(metadata["classes"]):
            raise ValueError(f"Invalid class ID for {r.source_id}")
        if r.condition == "clean":
            if r.severity != 0:
                raise ValueError("Clean rows must use severity 0")
            clean.add(r.source_id)
        elif r.condition not in CONDITIONS or r.severity not in range(1, 6):
            raise ValueError(f"Unknown condition/severity: {r.condition}/{r.severity}")
        old = sources.setdefault(r.source_id, (r.split, r.class_id))
        if old != (r.split, r.class_id):
            raise ValueError(f"Cross-split or conflicting-label source: {r.source_id}")
        k = (r.source_id, r.condition, r.severity)
        if k in keys:
            raise ValueError(f"Duplicate source-condition-severity: {k}")
        keys.add(k)
        p = (Path(root or ".") / r.path).resolve()
        if p in paths and paths[p] != k:
            raise ValueError(f"One image path is assigned multiple observation identities: {p}")
        paths[p] = k
        if check_files and not p.is_file():
            raise FileNotFoundError(p)
    if set(sources) != clean:
        raise ValueError("Every source must have exactly one clean reference")
    if not rows:
        raise ValueError("Empty manifest")


def stable_seed(*values):
    b = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(b).digest()[:8], "big")


def normalized(image, size):
    image = TF.resize(image, [size, size], interpolation=InterpolationMode.BICUBIC, antialias=True)
    return TF.normalize(TF.to_tensor(image), MEAN, STD)


def paired_transform(a, b, cfg, rng):
    if a.size != b.size:
        raise ValueError(f"Paired images must share geometry, got {a.size} and {b.size}; re-export aligned views")
    w, h = a.size
    area = h * w
    ratio = tuple(cfg["crop_ratio"])
    crop = None
    for _ in range(10):
        target = area * rng.uniform(*cfg["crop_scale"])
        aspect = math.exp(rng.uniform(math.log(ratio[0]), math.log(ratio[1])))
        cw, ch = round(math.sqrt(target * aspect)), round(math.sqrt(target / aspect))
        if 0 < cw <= w and 0 < ch <= h:
            crop = (rng.randrange(h - ch + 1), rng.randrange(w - cw + 1), ch, cw)
            break
    if crop is None:
        if w / h < ratio[0]:
            cw, ch = w, round(w / ratio[0])
        elif w / h > ratio[1]:
            cw, ch = round(h * ratio[1]), h
        else:
            cw, ch = w, h
        crop = ((h - ch) // 2, (w - cw) // 2, ch, cw)
    result = [TF.resized_crop(x, *crop, [cfg["image_size"]] * 2,
                              InterpolationMode.BICUBIC, antialias=True) for x in (a, b)]
    if rng.random() < cfg["horizontal_flip"]:
        result = [TF.hflip(x) for x in result]
    return tuple(TF.normalize(TF.to_tensor(x), MEAN, STD) for x in result)


class PairedDataset(Dataset):
    def __init__(self, rows, metadata, root, source_ids, conditions, levels, cfg, seed):
        self.root, self.cfg, self.seed, self.epoch = Path(root), cfg, seed, 0
        self.parents = [c["group_id"] for c in metadata["classes"]]
        wanted = set(source_ids)
        self.clean = sorted([r for r in rows if r.source_id in wanted and r.condition == "clean"], key=lambda r: r.source_id)
        self.views = {(r.source_id, r.condition, r.severity): r for r in rows if r.source_id in wanted}
        self.conditions, self.levels = list(conditions), list(levels)
        if not self.clean or not self.conditions or not self.levels:
            raise ValueError("Training source/condition/severity set is empty")
        for r in self.clean:
            for c in self.conditions:
                for level in self.levels:
                    if (r.source_id, c, level) not in self.views:
                        raise ValueError(f"Missing training view: {r.source_id}/{c}/{level}")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, index):
        a = self.clean[index]
        rng = random.Random(stable_seed(self.seed, self.epoch, a.source_id))
        b = self.views[(a.source_id, rng.choice(self.conditions), rng.choice(self.levels))]
        with Image.open(self.root / a.path) as im:
            image_a = im.convert("RGB")
        with Image.open(self.root / b.path) as im:
            image_b = im.convert("RGB")
        xa, xb = paired_transform(image_a, image_b, self.cfg, rng)
        return xa, xb, a.class_id, self.parents[a.class_id]


class EvaluationDataset(Dataset):
    def __init__(self, rows, root, size):
        self.rows = sorted(rows, key=lambda r: (r.condition, r.severity, r.source_id))
        self.root, self.size = Path(root), size
        if not self.rows:
            raise ValueError("Empty evaluation subset")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        with Image.open(self.root / r.path) as im:
            x = normalized(im.convert("RGB"), self.size)
        return x, r.class_id, index


def select_evaluation(rows, split, source_ids, conditions, levels, require_grid=True):
    wanted = set(source_ids)
    result = [r for r in rows if r.split == split and r.source_id in wanted and
              (r.condition == "clean" or (r.condition in conditions and r.severity in levels))]
    if require_grid:
        keys = {(r.source_id, r.condition, r.severity) for r in result}
        for source in wanted:
            for condition, severity in [("clean", 0)] + [(c, s) for c in conditions for s in levels]:
                if (source, condition, severity) not in keys:
                    raise ValueError(f"Incomplete evaluation grid: {source}/{condition}/{severity}")
    return result
