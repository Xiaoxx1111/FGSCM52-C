from collections import defaultdict
import numpy as np


def classification_metrics(probs, labels, candidate_ids, parents, bins=15):
    """Percent accuracies/ECE/F1; unscaled NLL, multiclass-sum Brier and AURC."""
    probs = np.asarray(probs, dtype=np.float64)
    labels, candidates = np.asarray(labels), np.asarray(candidate_ids)
    if probs.shape != (len(labels), len(candidates)) or len(labels) == 0:
        raise ValueError("Prediction shape or support is invalid")
    if not np.isfinite(probs).all() or not np.allclose(probs.sum(-1), 1, atol=1e-5):
        raise ValueError("Invalid probability distribution")
    inverse = {int(c): i for i, c in enumerate(candidates)}
    if not set(map(int, labels)) <= set(inverse):
        raise ValueError("Evaluation targets are outside the candidate set")
    target = np.array([inverse[int(c)] for c in labels])
    local = probs.argmax(-1)
    prediction = candidates[local]
    correct = (prediction == labels)
    confidence = probs.max(-1)
    confusion = np.zeros((len(candidates), len(candidates)), dtype=np.int64)
    np.add.at(confusion, (target, local), 1)
    tp = np.diag(confusion)
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(len(tp), dtype=float), where=denominator > 0)
    ece = 0.
    membership = np.minimum((confidence * bins).astype(int), bins - 1)
    for b in range(bins):
        mask = membership == b
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), target] = 1
    # Stable ordering makes confidence ties deterministic for source-sorted records.
    order = np.argsort(-confidence, kind="stable")
    risk = np.cumsum(~correct[order]) / np.arange(1, len(labels) + 1)
    parents = np.asarray(parents)
    return {"n": int(len(labels)), "top1": float(correct.mean() * 100),
            "macro_f1": float(f1.mean() * 100), "coarse_top1": float((parents[prediction] == parents[labels]).mean() * 100),
            "nll": float(-np.log(probs[np.arange(len(labels)), target].clip(1e-12)).mean()),
            "ece": float(ece * 100), "brier": float(((probs - onehot) ** 2).sum(-1).mean()),
            "aurc": float(risk.mean())}


def mean_metrics(items):
    return {k: float(np.mean([r[k] for r in items])) for k in items[0] if k != "n"}


def summarize_predictions(probs, labels, records, candidates, parents, conditions, levels, bins=15):
    groups = defaultdict(list)
    for i, r in enumerate(records):
        groups[(r.condition, r.severity)].append(i)
    cells = {}
    for condition in ["clean"] + list(conditions):
        for severity in ([0] if condition == "clean" else levels):
            indices = groups.get((condition, severity), [])
            if not indices:
                raise ValueError(f"Missing metric cell: {condition}/{severity}")
            ix = np.asarray(indices)
            cells[f"{condition}/{severity}"] = classification_metrics(probs[ix], labels[ix], candidates, parents, bins)
    per_condition = {c: mean_metrics([cells[f"{c}/{s}"] for s in levels]) for c in conditions}
    condition_mean = mean_metrics(list(per_condition.values()))
    return {"clean": cells["clean/0"], "mCA": condition_mean["top1"],
            "WCA": min(x["top1"] for x in per_condition.values()),
            "worst_condition": min(per_condition, key=lambda c: per_condition[c]["top1"]),
            "mean_condition_metrics": condition_mean, "per_condition": per_condition, "cells": cells}
