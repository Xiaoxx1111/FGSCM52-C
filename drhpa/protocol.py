import json
import random
from collections import defaultdict
from pathlib import Path
from .data import CONDITIONS, HOLDOUTS


def make_protocol(rows, metadata, cfg, seed):
    p = cfg["protocol"]
    mode = p["mode"]
    if mode not in {"closed_set", "held_out", "cross_severity", "base_to_new"}:
        raise ValueError(f"Unknown protocol: {mode}")
    if p.get("evaluation_candidates", "disjoint") not in {"disjoint", "joint"}:
        raise ValueError("evaluation_candidates must be disjoint or joint")
    train_conditions = list(CONDITIONS)
    held = []
    if mode == "held_out":
        held = list(HOLDOUTS[p["holdout_group"]])
        train_conditions = [c for c in CONDITIONS if c not in held]
    train_levels = list(p["train_levels"])
    val_levels = list(p["val_levels"])
    test_levels = list(p["test_levels"])
    if not set(val_levels) <= set(train_levels):
        raise ValueError("Validation severity levels must be visible during training")
    if any(x not in range(1, 6) for x in train_levels + val_levels + test_levels):
        raise ValueError("Severity levels must lie in 1..5")
    all_ids = list(range(len(metadata["classes"])))
    base, novel = all_ids, []
    if mode == "base_to_new":
        if not p.get("base_class_ids"):
            raise ValueError("base_to_new requires a JSON file of actual base class IDs; no class split is inferred")
        base = json.loads(Path(p["base_class_ids"]).read_text())
        if not isinstance(base, list) or len(base) != len(set(base)) or not set(base) < set(all_ids):
            raise ValueError("Base class IDs must be a nonempty strict subset without duplicates")
        base = sorted(base)
        novel = [c for c in all_ids if c not in base]
        if not base:
            raise ValueError("Base class set is empty")
    source_sets = {s: {r.source_id for r in rows if r.split == s and r.condition == "clean"} for s in ["train", "val", "test"]}
    by_class = defaultdict(list)
    for r in rows:
        if r.split == "train" and r.condition == "clean" and r.class_id in base:
            by_class[r.class_id].append(r.source_id)
    train_ids = []
    rng = random.Random(seed)
    for c in base:
        ids = sorted(by_class[c])
        if not ids:
            raise ValueError(f"No training source for class {c}")
        if mode == "base_to_new":
            shots = int(p["shots"])
            if shots < 1 or len(ids) < shots:
                raise ValueError(f"Insufficient source images for {shots}-shot class {c}")
            # One fixed permutation per class; increasing K yields nested K-shot sets.
            rng.shuffle(ids)
            ids = ids[:shots]
        train_ids.extend(ids)
    val_ids = {r.source_id for r in rows if r.split == "val" and r.condition == "clean" and r.class_id in base}
    for c in base:
        if not any(r.class_id == c and r.source_id in val_ids for r in rows if r.condition == "clean"):
            raise ValueError(f"No base validation source for class {c}")
    if not val_ids or not source_sets["test"]:
        raise ValueError("Validation and test source sets must be nonempty")
    return {"mode": mode, "train_conditions": train_conditions, "val_conditions": train_conditions,
            "test_conditions": CONDITIONS, "held_out_conditions": held,
            "train_levels": train_levels, "val_levels": val_levels, "test_levels": test_levels,
            "train_source_ids": sorted(train_ids), "val_source_ids": sorted(val_ids),
            "test_source_ids": sorted(source_sets["test"]), "base_ids": base, "novel_ids": novel,
            "training_candidate_ids": base, "evaluation_candidates": p.get("evaluation_candidates", "disjoint")}
