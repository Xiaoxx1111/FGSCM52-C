import torch
from torch.nn import functional as F


def paired_loss(output, fine_labels, coarse_labels, cfg):
    """Batch layout is [N clean, N transformed]; all supervised CE uses 2N mean."""
    n = len(fine_labels)
    if output["logits"].shape[0] != 2 * n:
        raise ValueError("Expected clean then transformed images for each pair batch")
    candidates = output["candidate_ids"]
    inverse = torch.full((int(candidates.max()) + 1,), -1, device=candidates.device, dtype=torch.long)
    inverse[candidates] = torch.arange(len(candidates), device=candidates.device)
    if fine_labels.max() >= len(inverse) or (inverse[fine_labels] < 0).any():
        raise ValueError("Training labels fall outside the selected base-class candidates")
    target = inverse[fine_labels].repeat(2)
    fine_ce = F.cross_entropy(output["logits"], target)
    coarse_ce = F.cross_entropy(output["coarse_logits"], coarse_labels.repeat(2))
    sup = fine_ce + cfg["coarse_weight"] * coarse_ce
    tc = cfg["consistency_temperature"]
    if tc <= 0:
        raise ValueError("Consistency temperature must be positive")
    clean = (output["logits"][:n] / tc).softmax(-1).detach()
    distorted = (output["logits"][n:] / tc).log_softmax(-1)
    consistency = F.kl_div(distorted, clean, reduction="batchmean") * tc * tc
    q = output["q"].float()
    mean = q.mean(0)
    log_q = q.clamp_min(1e-8).log()
    balance = (mean * mean.clamp_min(1e-8).log()).sum()
    image_entropy = -(q * log_q).sum(-1).mean()
    routing = balance + cfg["entropy_gamma"] * image_entropy
    total = sup + cfg["consistency_weight"] * consistency + cfg["balance_weight"] * routing
    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite training objective")
    return total, {"loss": total.detach(), "fine_ce": fine_ce.detach(), "coarse_ce": coarse_ce.detach(),
                   "consistency": consistency.detach(), "routing": routing.detach(),
                   "routing_entropy": image_entropy.detach(), "top_expert_mass": q.max(-1).values.mean().detach()}
