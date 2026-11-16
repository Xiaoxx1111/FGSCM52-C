from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import PairedDataset, EvaluationDataset, read_metadata, read_manifest, select_evaluation
from .losses import paired_loss
from .metrics import summarize_predictions
from .model import DRHPA, load_backbone
from .protocol import make_protocol


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def seed_everything(seed, deterministic):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def device_for(name):
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu for checks")
    return torch.device(name)


def autocast_for(device, amp):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(amp and device.type == "cuda"))


def learning_rate(epoch, cfg):
    warm = cfg["warmup_epochs"]
    if epoch < warm:
        return cfg["warmup_lr"]
    n = cfg["epochs"] - warm
    progress = (epoch - warm) / max(n - 1, 1)
    return cfg["min_lr"] + (cfg["lr"] - cfg["min_lr"]) * (1 + math.cos(math.pi * progress)) / 2


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(2 ** 20), b""):
            h.update(block)
    return h.hexdigest()


def make_optimizer(model, cfg):
    # Weight decay only on router matrix weights, not contexts, keys, biases or log-scales.
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (decay if name.startswith("router.") and p.ndim > 1 else no_decay).append(p)
    return torch.optim.SGD([{"params": decay, "weight_decay": cfg["weight_decay"]},
                            {"params": no_decay, "weight_decay": 0.}],
                           lr=cfg["lr"], momentum=cfg["momentum"])


def loader(dataset, batch_size, workers, shuffle=False, seed=0, cuda=False):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=cuda, drop_last=False, persistent_workers=False, generator=g)


class Predictor:
    def __init__(self, dataset, cfg, device, cache_features=False):
        self.dataset, self.cfg, self.device = dataset, cfg, device
        self.cache_features, self.features = cache_features, None

    @torch.no_grad()
    def predict(self, model, candidates):
        model.eval()
        with autocast_for(self.device, self.cfg["train"]["amp"]):
            model.cache_text()
        n = len(self.dataset)
        probs = np.empty((n, len(candidates)), dtype=np.float32)
        labels = np.array([r.class_id for r in self.dataset.rows], dtype=np.int64)
        routing = np.empty((n, model.experts), dtype=np.float32)
        new_features = []
        if self.features is None:
            batches = loader(self.dataset, self.cfg["train"]["eval_batch_size"], self.cfg["train"]["workers"],
                             cuda=self.device.type == "cuda")
            for images, _, index in batches:
                with autocast_for(self.device, self.cfg["train"]["amp"]):
                    z = model.image_features(images.to(self.device, non_blocking=True))
                    out = model.forward_features(z, candidates)
                probs[index.numpy()] = out["logits"].softmax(-1).cpu().numpy()
                routing[index.numpy()] = out["q"].cpu().numpy()
                if self.cache_features:
                    new_features.append(z.cpu())
            if self.cache_features:
                self.features = torch.cat(new_features)
        else:
            bs = self.cfg["train"]["eval_batch_size"]
            for start in range(0, n, bs):
                z = self.features[start:start + bs].to(self.device)
                with autocast_for(self.device, self.cfg["train"]["amp"]):
                    out = model.forward_features(z, candidates)
                probs[start:start + bs] = out["logits"].softmax(-1).cpu().numpy()
                routing[start:start + bs] = out["q"].cpu().numpy()
        return probs, labels, routing


def metric_summary(prediction, dataset, model, candidates, cfg, conditions, levels):
    return summarize_predictions(prediction[0], prediction[1], dataset.rows, candidates,
                                 model.parents.cpu().numpy(), conditions, levels, cfg["evaluation"]["ece_bins"])


def save_predictions(path, prediction, dataset, candidates, metadata):
    probs, labels, routing = prediction
    path = Path(path)
    np.savez_compressed(path.with_suffix(".npz"), probabilities=probs, labels=labels, routing=routing,
                        candidate_ids=np.array(candidates), source_ids=np.array([r.source_id for r in dataset.rows]),
                        conditions=np.array([r.condition for r in dataset.rows]),
                        severities=np.array([r.severity for r in dataset.rows]))
    pred = np.array(candidates)[probs.argmax(-1)]
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_id", "condition", "severity", "class_id", "predicted_id", "confidence"] + [f"q_{m}" for m in range(routing.shape[1])])
        for i, r in enumerate(dataset.rows):
            writer.writerow([r.source_id, r.condition, r.severity, r.class_id, int(pred[i]), float(probs[i].max()), *routing[i].tolist()])
    diagnostics = {}
    for condition in sorted({r.condition for r in dataset.rows}):
        mask = np.array([r.condition == condition for r in dataset.rows])
        q = routing[mask]
        diagnostics[condition] = {"n": int(mask.sum()), "mean_weights": q.mean(0).tolist(),
                                  "mean_entropy": float(-(q * np.log(q.clip(1e-8))).sum(-1).mean()),
                                  "mean_top_expert_mass": float(q.max(-1).mean())}
    write_json(path.parent / (path.stem + "_routing.json"), diagnostics)
    condition_array = np.array([r.condition for r in dataset.rows])
    severity_array = np.array([r.severity for r in dataset.rows])
    condition_names = sorted(set(condition_array) - {"clean"})
    # Each class's worst condition is min_k mean_s recall, with severity levels equally weighted.
    with (path.parent / (path.stem + "_class_recall.csv")).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "clean_support", "clean_recall", "worst_condition", "worst_mean_recall", "decline_pp"])
        for c in candidates:
            class_mask = labels == c
            clean = class_mask & (condition_array == "clean")
            if not clean.any():
                continue
            clean_recall = float((pred[clean] == c).mean() * 100)
            scores = {}
            for condition in condition_names:
                level_scores = []
                for level in sorted(set(severity_array[condition_array == condition])):
                    mask = class_mask & (condition_array == condition) & (severity_array == level)
                    if mask.any():
                        level_scores.append(float((pred[mask] == c).mean() * 100))
                if level_scores:
                    scores[condition] = float(np.mean(level_scores))
            if scores:
                worst = min(scores, key=scores.get)
                writer.writerow([c, metadata["classes"][c]["name"], int(clean.sum()), clean_recall,
                                 worst, scores[worst], clean_recall - scores[worst]])


def train(cfg, resume=None, backbone_factory=load_backbone):
    out_dir = Path(cfg["output"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "last.pt").exists() and resume is None:
        raise FileExistsError("Run directory already has a checkpoint; use --resume or a new --output")
    device = device_for(cfg["device"])
    seed_everything(cfg["seed"], cfg["train"]["deterministic"])
    metadata = read_metadata(cfg["data"]["metadata"])
    records = read_manifest(cfg["data"]["manifest"], metadata, cfg["data"]["check_files"], cfg["data"]["root"])
    protocol = make_protocol(records, metadata, cfg, cfg["seed"])
    dataset = PairedDataset(records, metadata, cfg["data"]["root"], protocol["train_source_ids"],
                            protocol["train_conditions"], protocol["train_levels"], cfg["data"], cfg["seed"])
    val_rows = select_evaluation(records, "val", protocol["val_source_ids"], protocol["val_conditions"], protocol["val_levels"])
    val_data = EvaluationDataset(val_rows, cfg["data"]["root"], cfg["data"]["image_size"])
    backbone, tokenize = backbone_factory(cfg["model"], device)
    model = DRHPA(backbone, tokenize, metadata, cfg["model"]).to(device)
    optimizer = make_optimizer(model, cfg["train"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["amp"] and device.type == "cuda")
    signature = {"manifest": fingerprint(cfg["data"]["manifest"]), "metadata": fingerprint(cfg["data"]["metadata"])}
    start, best = 0, (-float("inf"), -float("inf"))
    if resume:
        # Load only a trusted checkpoint created by this implementation.
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        comparable = lambda x: {k: v for k, v in x.items() if k not in {"output", "device"}}
        if saved["signature"] != signature or comparable(saved["config"]) != comparable(cfg):
            raise ValueError("Resume requires identical data and experiment configuration")
        model.load_adapter(saved["adapter"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        start, best = saved["epoch"] + 1, tuple(saved["best"])
        random.setstate(saved["rng"]["python"])
        np.random.set_state(saved["rng"]["numpy"])
        torch.set_rng_state(saved["rng"]["torch"])
        if device.type == "cuda" and saved["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
    write_json(out_dir / "config.json", cfg)
    write_json(out_dir / "protocol.json", protocol)
    write_json(out_dir / "metadata.json", metadata)
    write_json(out_dir / "environment.json", {"python": platform.python_version(), "torch": torch.__version__,
               "cuda": torch.version.cuda, "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               "dataset_sha256": signature, "parameters": model.parameter_counts(),
               "configuration_origin": "new_design_not_recovered_original"})
    evaluator = Predictor(val_data, cfg, device, cfg["train"]["validation_feature_cache"])
    for epoch in range(start, cfg["train"]["epochs"]):
        begin = time.perf_counter()
        dataset.set_epoch(epoch)
        model.train()
        lr = learning_rate(epoch, cfg["train"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        totals, count = {}, 0
        for clean, transformed, fine, coarse in loader(dataset, cfg["train"]["batch_pairs"], cfg["train"]["workers"],
                                                      shuffle=True, seed=cfg["seed"] + epoch, cuda=device.type == "cuda"):
            n = len(fine)
            images = torch.cat([clean, transformed]).to(device, non_blocking=True)
            fine, coarse = fine.to(device), coarse.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_for(device, cfg["train"]["amp"]):
                prediction = model(images, protocol["training_candidate_ids"])
                loss, parts = paired_loss(prediction, fine, coarse, cfg["loss"])
            scaler.scale(loss).backward()
            if cfg["train"]["gradient_clip"] is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg["train"]["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            model.clamp_scales()
            for k, v in parts.items():
                totals[k] = totals.get(k, 0.) + float(v) * n
            count += n
        pred = evaluator.predict(model, protocol["base_ids"])
        val = metric_summary(pred, val_data, model, protocol["base_ids"], cfg, protocol["val_conditions"], protocol["val_levels"])
        score = (val["mCA"], val["clean"]["top1"])
        improved = score > best
        if improved:
            best = score
        info = {"epoch": epoch + 1, "lr": lr, "seconds": time.perf_counter() - begin,
                **{k: v / count for k, v in totals.items()}, "val_mCA": val["mCA"], "val_clean": val["clean"]["top1"], "val_WCA": val["WCA"]}
        with (out_dir / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(info) + "\n")
        print(json.dumps(info), flush=True)
        state = {"adapter": model.adapter_state(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                 "epoch": epoch, "best": best, "config": cfg, "metadata": metadata, "protocol": protocol, "signature": signature,
                 "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                         "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}}
        torch.save(state, out_dir / "last.tmp")
        (out_dir / "last.tmp").replace(out_dir / "last.pt")
        if improved:
            torch.save(state, out_dir / "best.tmp")
            (out_dir / "best.tmp").replace(out_dir / "best.pt")
            write_json(out_dir / "best_validation.json", val)
    return model


def evaluate(checkpoint_path, output, device_name=None, backbone_factory=load_backbone):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg, metadata, protocol = saved["config"], saved["metadata"], saved["protocol"]
    device = device_for(device_name or cfg["device"])
    records = read_manifest(cfg["data"]["manifest"], metadata, cfg["data"]["check_files"], cfg["data"]["root"])
    if fingerprint(cfg["data"]["manifest"]) != saved["signature"]["manifest"]:
        raise ValueError("Manifest changed after training; use the recorded manifest")
    backbone, tokenize = backbone_factory(cfg["model"], device)
    model = DRHPA(backbone, tokenize, metadata, cfg["model"]).to(device)
    model.load_adapter(saved["adapter"])
    rows = select_evaluation(records, "test", protocol["test_source_ids"], protocol["test_conditions"], protocol["test_levels"])
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    def run_subset(name, subset, candidates):
        ds = EvaluationDataset(subset, cfg["data"]["root"], cfg["data"]["image_size"])
        pred = Predictor(ds, cfg, device).predict(model, candidates)
        result = metric_summary(pred, ds, model, candidates, cfg, protocol["test_conditions"], protocol["test_levels"])
        if cfg["evaluation"]["export_probabilities"]:
            save_predictions(out_dir / name, pred, ds, candidates, metadata)
        return result
    if protocol["mode"] == "base_to_new":
        all_ids = list(range(len(metadata["classes"])))
        base = run_subset("base", [r for r in rows if r.class_id in protocol["base_ids"]],
                          all_ids if protocol["evaluation_candidates"] == "joint" else protocol["base_ids"])
        novel = run_subset("novel", [r for r in rows if r.class_id in protocol["novel_ids"]],
                           all_ids if protocol["evaluation_candidates"] == "joint" else protocol["novel_ids"])
        a, b = base["mCA"], novel["mCA"]
        result = {"base": base, "novel": novel, "H_B2N": 2 * a * b / (a + b) if a + b else 0.,
                  "aggregation": "harmonic_mean_of_base_and_novel_mCA_per_seed", "candidate_mode": protocol["evaluation_candidates"]}
    else:
        result = run_subset("test", rows, list(range(len(metadata["classes"]))))
        if protocol["mode"] == "held_out":
            pc = result["per_condition"]
            result["Seen"] = float(np.mean([pc[c]["top1"] for c in protocol["train_conditions"]]))
            result["Unseen"] = float(np.mean([pc[c]["top1"] for c in protocol["held_out_conditions"]]))
            result["Gap"] = result["Seen"] - result["Unseen"]
    write_json(out_dir / "metrics.json", result)
    return result
