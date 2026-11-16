"""Frozen OpenAI CLIP with differentiable text contexts and score-level experts."""
from __future__ import annotations

import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class PromptBank(nn.Module):
    def __init__(self, clip_model, tokenize, names, experts, n_ctx, init_std):
        super().__init__()
        self.experts, self.n_ctx = experts, n_ctx
        length = clip_model.context_length
        # X is one BPE token; verify this rather than assuming a tokenizer layout.
        if tokenize("X")[0].argmax().item() != 2:
            raise ValueError("Expected the OpenAI CLIP tokenizer and one-token X prefix")
        texts = [" ".join(["X"] * n_ctx) + " " + name.replace("_", " ") + "." for name in names]
        tokens = tokenize(texts, context_length=length, truncate=False)
        with torch.no_grad():
            fixed = clip_model.token_embedding(tokens.to(clip_model.token_embedding.weight.device)).float()
        self.register_buffer("prefix", fixed[:, :1].clone())
        self.register_buffer("suffix", fixed[:, 1 + n_ctx:].clone())
        self.register_buffer("eot", tokens.argmax(dim=-1).to(fixed.device))
        self.context = nn.Parameter(torch.empty(experts, n_ctx, fixed.shape[-1], device=fixed.device))
        nn.init.normal_(self.context, std=init_std)

    def sequences(self):
        m, c = self.experts, self.prefix.shape[0]
        return torch.cat([
            self.prefix.unsqueeze(0).expand(m, -1, -1, -1),
            self.context[:, None].expand(-1, c, -1, -1),
            self.suffix.unsqueeze(0).expand(m, -1, -1, -1),
        ], dim=2).reshape(m * c, -1, self.context.shape[-1])


class DRHPA(nn.Module):
    def __init__(self, clip_model, tokenize, metadata, cfg):
        super().__init__()
        self.clip = clip_model.float().eval()
        for p in self.clip.parameters():
            p.requires_grad_(False)
        if not hasattr(self.clip.visual, "class_embedding"):
            raise ValueError("This implementation supports OpenAI CLIP ViT backbones")
        self.cfg = dict(cfg)
        self.experts = int(cfg["experts"])
        d_in = self.clip.text_projection.shape[1]
        self.fine = PromptBank(self.clip, tokenize, [x["name"] for x in metadata["classes"]],
                               self.experts, cfg["context_length"], cfg["context_init_std"])
        self.use_hierarchy = bool(cfg.get("hierarchy", True))
        self.group_count = len(metadata["groups"])
        self.coarse = (PromptBank(self.clip, tokenize, metadata["groups"],
                                  self.experts, cfg["context_length"], cfg["context_init_std"])
                       if self.use_hierarchy else None)
        self.register_buffer("parents", torch.tensor([x["group_id"] for x in metadata["classes"]],
                                                     dtype=torch.long))
        self.mixing = cfg.get("mixing", "learned")
        if self.mixing not in {"learned", "uniform"}:
            raise ValueError("mixing must be learned or uniform")
        self.router = (nn.Sequential(nn.Linear(d_in, cfg["router_hidden"]), nn.GELU(),
                                     nn.Linear(cfg["router_hidden"], cfg["router_dim"]))
                       if self.mixing == "learned" else None)
        self.keys = (nn.Parameter(torch.randn(self.experts, cfg["router_dim"]) * .02)
                     if self.mixing == "learned" else None)
        self.log_fine_scale = nn.Parameter(torch.tensor(math.log(1 / cfg["fine_temperature"])),
                                           requires_grad=cfg["learn_score_temperatures"])
        self.log_coarse_scale = nn.Parameter(torch.tensor(math.log(1 / cfg["coarse_temperature"])),
                                             requires_grad=cfg["learn_score_temperatures"] and self.use_hierarchy)
        self.routing_temperature = float(cfg["routing_temperature"])
        self.eta = float(cfg["eta"]) if self.use_hierarchy else 0.
        if self.routing_temperature <= 0 or self.eta < 0:
            raise ValueError("Routing temperature must be positive and eta nonnegative")
        self._text_cache = None

    def train(self, mode=True):
        super().train(mode)
        self.clip.eval()  # Frozen backbone stays deterministic even while prompts train.
        self._text_cache = None
        return self

    def _apply(self, fn):
        self._text_cache = None
        return super()._apply(fn)

    def encode_prompt(self, sequences, eot):
        # DO NOT wrap this in no_grad: the inputs are trainable context vectors.
        x = sequences + self.clip.positional_embedding.to(sequences.dtype)
        x = self.clip.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.clip.ln_final(x)
        x = x[torch.arange(len(x), device=x.device), eot] @ self.clip.text_projection
        return F.normalize(x.float(), dim=-1, eps=1e-6)

    def encode_bank(self, bank):
        seq = bank.sequences()
        eot = bank.eot.repeat(self.experts)
        chunks = []
        size = int(self.cfg["text_chunk_size"])
        for start in range(0, len(seq), size):
            a, b = seq[start:start + size], eot[start:start + size]
            if self.training and self.cfg["text_gradient_checkpointing"]:
                chunk = checkpoint(self.encode_prompt, a, b, use_reentrant=False)
            else:
                chunk = self.encode_prompt(a, b)
            chunks.append(chunk)
        return torch.cat(chunks).reshape(self.experts, len(bank.eot), -1)

    def text_features(self):
        if not self.training and self._text_cache is not None:
            return self._text_cache
        return self.encode_bank(self.fine), self.encode_bank(self.coarse) if self.coarse is not None else None

    @torch.no_grad()
    def cache_text(self):
        if self.training:
            raise RuntimeError("Call eval() before caching learned text features")
        self._text_cache = (self.encode_bank(self.fine), self.encode_bank(self.coarse) if self.coarse is not None else None)

    @torch.no_grad()
    def image_features(self, images):
        v = self.clip.visual
        patch = v.conv1.kernel_size[0]
        if images.shape[-2] % patch or images.shape[-1] % patch:
            raise ValueError("Image dimensions must be divisible by the visual patch size")
        x = v.conv1(images.to(v.conv1.weight.dtype))
        h, w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        cls = v.class_embedding.to(x.dtype)[None, None].expand(len(x), 1, -1)
        x = torch.cat([cls, x], dim=1)
        pos = v.positional_embedding.float()
        grid = math.isqrt(pos.shape[0] - 1)
        if grid * grid != pos.shape[0] - 1:
            raise ValueError("Expected a square pretrained positional grid")
        if (h, w) != (grid, grid):
            spatial = pos[1:].reshape(grid, grid, -1).permute(2, 0, 1)[None]
            spatial = F.interpolate(spatial, size=(h, w), mode="bicubic", align_corners=False)
            pos = torch.cat([pos[:1], spatial[0].permute(1, 2, 0).reshape(h * w, -1)])
        x = v.ln_pre(x + pos.to(x.dtype))
        x = v.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = v.ln_post(x[:, 0])
        if v.proj is not None:
            x = x @ v.proj
        return F.normalize(x.float(), dim=-1, eps=1e-6)

    def forward(self, images, candidates=None):
        return self.forward_features(self.image_features(images), candidates)

    def forward_features(self, z, candidates=None):
        fine_t, coarse_t = self.text_features()
        # Keep routing and output probabilities in FP32 even under image/text AMP.
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            if self.mixing == "learned":
                r = F.normalize(self.router(z), dim=-1, eps=1e-6)
                keys = F.normalize(self.keys.float(), dim=-1, eps=1e-6)
                q = (r @ keys.T / self.routing_temperature).softmax(-1)
            else:
                q = z.new_full((len(z), self.experts), 1 / self.experts)
            lo, hi = math.log(self.cfg["score_scale_min"]), math.log(self.cfg["score_scale_max"])
            fine = torch.einsum("bd,mcd,bm->bc", z, fine_t.float(), q) * self.log_fine_scale.clamp(lo, hi).exp()
            coarse = (torch.einsum("bd,mhd,bm->bh", z, coarse_t.float(), q) * self.log_coarse_scale.clamp(lo, hi).exp()
                      if coarse_t is not None else z.new_zeros((len(z), self.group_count)))
            fused = fine + self.eta * coarse[:, self.parents]
            ids = torch.arange(fused.shape[1], device=fused.device) if candidates is None else torch.as_tensor(candidates, device=fused.device)
            return {"logits": fused[:, ids], "coarse_logits": coarse, "q": q, "candidate_ids": ids}

    @torch.no_grad()
    def clamp_scales(self):
        a, b = math.log(self.cfg["score_scale_min"]), math.log(self.cfg["score_scale_max"])
        self.log_fine_scale.clamp_(a, b)
        self.log_coarse_scale.clamp_(a, b)

    def adapter_state(self):
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if not k.startswith("clip.")}

    def load_adapter(self, state):
        expected = {k for k in self.state_dict() if not k.startswith("clip.")}
        if set(state) != expected:
            raise ValueError(f"Adapter keys differ: missing={expected-set(state)}, extra={set(state)-expected}")
        self.load_state_dict(state, strict=False)
        self._text_cache = None

    def parameter_counts(self):
        return {"total": sum(p.numel() for p in self.parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "contexts": self.fine.context.numel() + (self.coarse.context.numel() if self.coarse else 0),
                "router_and_keys": (sum(p.numel() for p in self.router.parameters()) + self.keys.numel()) if self.router else 0}


def load_backbone(cfg, device):
    import clip
    name = cfg.get("checkpoint") or cfg["backbone"]
    if cfg.get("checkpoint") and not Path(name).is_file():
        raise FileNotFoundError(f"CLIP checkpoint does not exist: {name}")
    model, _ = clip.load(name, device="cpu", jit=False, download_root=cfg.get("download_root"))
    return model.float().to(device), clip.tokenize
