"""
Roof Experiment — Testing KL Asymmetry and Endogenous Maintenance
==================================================================

Three experiments testing the Maintaining Divergence framework:

Experiment 1 — Distillation Direction:
  Forward KL (mode-covering), Reverse KL (mode-seeking), and Jensen-Shannon
  (symmetric) distillation from the same teacher. Tests whether the direction
  of knowledge transfer determines uncertainty calibration at unrewarded positions.

Experiment 2 — Non-Invertible Generating Process:
  Quadratic recurrence x_{t+1} = x_t^2 + b mod 17 (many-to-one) vs
  linear recurrence x_{t+1} = ax_t + b mod 17 (one-to-one). Forward vs
  backward prediction. Tests whether the statistical asymmetry created by
  genuine causation is detectable only in the causal direction.

Experiment 3 — Endogenous Roof:
  Learned temperature scaling and attention positional forget-gate. Tests
  whether the model can learn to pay its own synchronization tax — maintaining
  the causal circuit's calibration without external subsidy.

Usage:
    # Experiment 1: full matrix
    python roof_experiment.py --experiment distill_direction --run_matrix \
        --seeds 42 43 44 --device mps

    # Experiment 2: full matrix
    python roof_experiment.py --experiment causal_direction --run_matrix \
        --seeds 42 43 44 --device mps

    # Experiment 3: single endogenous condition
    python roof_experiment.py --experiment endogenous_roof \
        --endogenous temperature --seeds 42 --device mps

    # Experiment 3: full matrix
    python roof_experiment.py --experiment endogenous_roof --run_matrix \
        --seeds 42 43 44 --device mps
"""

import os
import math
import argparse
import json
import numpy as np

# Reuse from wall_erosion_experiment
from wall_erosion_experiment import (
    _ensure_torch, _resolve_device, _seed_all,
    _build_model_class, generate_batch,
    _masked_ce_loss, _entropy_from_logits,
    compute_distill_subsidy, compute_entropy_subsidy,
    compute_wall_metrics, compute_erosion_fraction,
)
from recurrence_bwt import (
    RecurrenceConfig,
    generate_recurrence_sequence,
    bayesian_predictive_recurrence,
    _predictive_entropy,
)
from recurrence_extrapolation import evaluate_at_length
from quadratic_recurrence import (
    QuadraticConfig, generate_quadratic_sequence,
    bayesian_predictive_quadratic,
    bayesian_predictive_quadratic_backward,
    bayesian_predictive_linear_backward,
)

# Lazy torch imports (populated by _ensure_torch)
torch = None
nn = None
F = None


def _sync_torch():
    """Sync module-level torch refs after _ensure_torch()."""
    global torch, nn, F
    import torch as _torch
    import torch.nn as _nn
    import torch.nn.functional as _F
    torch = _torch
    nn = _nn
    F = _F


# ============================================================================
# Endogenous roof model
# ============================================================================

def _build_endogenous_model_class(endogenous_type='none'):
    """Build a model with optional endogenous roof mechanisms.

    Args:
        endogenous_type: 'none', 'temperature', 'gate', or 'both'

    The model wraps RecurrenceTransformerSubsidy and adds:
    - Temperature head: MLP that outputs per-position temperature scalar,
      applied to final logits. The model learns when to sharpen/soften.
    - Positional forget-gate: separates content and positional streams in
      Q/K with a learned sigmoid gate that can suppress positional info.
    """
    _ensure_torch()
    _sync_torch()

    use_temp = endogenous_type in ('temperature', 'both')
    use_gate = endogenous_type in ('gate', 'both')

    class TemperatureHead(nn.Module):
        """Learned per-position temperature scaling."""
        def __init__(self, d_model):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, 1),
                nn.Softplus(),  # T > 0
            )
            # Initialize to output ~1.0 (no-op temperature)
            with torch.no_grad():
                self.net[-2].bias.fill_(0.55)  # softplus(0.55) ≈ 1.0

        def forward(self, hiddens):
            return self.net(hiddens).squeeze(-1)  # (B, T)

    class GatedMultiHeadAttention(nn.Module):
        """Attention with positional forget-gate.

        Separates content and positional streams in Q/K projections.
        A learned sigmoid gate controls how much positional information
        enters the attention computation. gate=0 means pure content
        attention (position-blind); gate=1 means full positional info.
        """
        def __init__(self, d_model, n_heads, dropout=0.1):
            super().__init__()
            self.n_heads = n_heads
            self.d_head = d_model // n_heads

            # Content QKV (applied to token embeddings only)
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.out_proj = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)

            # Positional gate: per-head sigmoid from content features
            self.pos_gate_proj = nn.Sequential(
                nn.Linear(d_model, n_heads),
                nn.Sigmoid(),
            )
            # Initialize gate bias high (~0.8) so the model starts
            # with mostly-on positional information
            with torch.no_grad():
                self.pos_gate_proj[0].bias.fill_(1.4)  # sigmoid(1.4) ≈ 0.8

        def forward(self, x, pos_embed, mask=None):
            """Forward with gated positional embedding.

            Args:
                x: (B, T, d_model) — token embeddings + (gated) pos embeddings
                pos_embed: (B, T, d_model) — raw positional embeddings
                mask: causal mask
            Returns:
                output: (B, T, d_model)
                alpha: (B, n_heads, T, T) attention weights
                gate_values: (B, T, n_heads) gate activations for diagnostics
            """
            B, T, C = x.shape

            # Compute gate from the input (which already has some pos info)
            gate = self.pos_gate_proj(x)  # (B, T, n_heads)

            # Compute QKV from content
            qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
            q, k, v = qkv.unbind(dim=2)

            # Compute positional QK contribution
            qkv_pos = self.qkv(pos_embed).reshape(
                B, T, 3, self.n_heads, self.d_head)
            q_pos, k_pos, _ = qkv_pos.unbind(dim=2)

            # Gate the positional contribution: (B, T, n_heads, 1)
            gate_expanded = gate.unsqueeze(-1)  # (B, T, n_heads, 1)
            q = q + gate_expanded * q_pos
            k = k + gate_expanded * k_pos

            q = q.transpose(1, 2)  # (B, n_heads, T, d_head)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            scale = self.d_head ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            if mask is not None:
                attn = attn.masked_fill(
                    mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            alpha = torch.softmax(attn, dim=-1)
            alpha = self.dropout(alpha)
            out = (alpha @ v).transpose(1, 2).reshape(B, T, C)

            return self.out_proj(out), alpha, gate

    class GatedTransformerBlock(nn.Module):
        """Transformer block with optional positional forget-gate."""
        def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
            super().__init__()
            d_ff = d_ff or 4 * d_model
            self.attn = GatedMultiHeadAttention(d_model, n_heads, dropout)
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )

        def forward(self, x, pos_embed, mask=None):
            h, alpha, gate = self.attn(self.ln1(x), pos_embed, mask)
            x = x + h
            x = x + self.ff(self.ln2(x))
            return x, alpha, gate

    class RecurrenceTransformerEndogenous(nn.Module):
        """Transformer with endogenous roof mechanisms.

        Supports:
        - Temperature head: learned per-position temperature for logit scaling
        - Positional forget-gate: learned per-head gate on positional info
        - Both simultaneously
        """
        def __init__(self, vocab_size, n_tokens, d_model=192, n_layers=6,
                     n_heads=6, d_ff=768, dropout=0.1):
            super().__init__()
            self.vocab_size = vocab_size
            self.n_tokens = n_tokens
            self.d_model = d_model
            self.use_temp = use_temp
            self.use_gate = use_gate

            self.token_embed = nn.Embedding(
                vocab_size + 1, d_model, padding_idx=vocab_size)
            self.pos_embed = nn.Embedding(512, d_model)

            if use_gate:
                self.layers = nn.ModuleList([
                    GatedTransformerBlock(d_model, n_heads, d_ff, dropout)
                    for _ in range(n_layers)
                ])
            else:
                # Use standard blocks from wall_erosion_experiment
                from wall_erosion_experiment import _build_model_class
                # We need the TransformerBlock class — rebuild it
                _base = _build_model_class()
                # Actually, just inline standard attention
                self.layers = nn.ModuleList([
                    self._make_standard_block(d_model, n_heads, d_ff, dropout)
                    for _ in range(n_layers)
                ])

            self.ln_final = nn.LayerNorm(d_model)
            self.output_proj = nn.Linear(d_model, n_tokens)

            # Temperature head
            self.temp_head = None
            if use_temp:
                self.temp_head = TemperatureHead(d_model)

        def _make_standard_block(self, d_model, n_heads, d_ff, dropout):
            """Build a standard transformer block (no gate)."""

            class _StdMultiHeadAttention(nn.Module):
                def __init__(self_inner):
                    super().__init__()
                    self_inner.n_heads = n_heads
                    self_inner.d_head = d_model // n_heads
                    self_inner.qkv = nn.Linear(d_model, 3 * d_model)
                    self_inner.out_proj = nn.Linear(d_model, d_model)
                    self_inner.dropout_layer = nn.Dropout(dropout)

                def forward(self_inner, x, mask=None):
                    B, T, C = x.shape
                    qkv = self_inner.qkv(x).reshape(
                        B, T, 3, self_inner.n_heads, self_inner.d_head)
                    q, k, v = qkv.unbind(dim=2)
                    q = q.transpose(1, 2)
                    k = k.transpose(1, 2)
                    v = v.transpose(1, 2)
                    scale = self_inner.d_head ** -0.5
                    attn = (q @ k.transpose(-2, -1)) * scale
                    if mask is not None:
                        attn = attn.masked_fill(
                            mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                    alpha = torch.softmax(attn, dim=-1)
                    alpha = self_inner.dropout_layer(alpha)
                    out = (alpha @ v).transpose(1, 2).reshape(B, T, C)
                    return self_inner.out_proj(out), alpha

            class _StdBlock(nn.Module):
                def __init__(self_inner):
                    super().__init__()
                    self_inner.attn = _StdMultiHeadAttention()
                    self_inner.ln1 = nn.LayerNorm(d_model)
                    self_inner.ln2 = nn.LayerNorm(d_model)
                    self_inner.ff = nn.Sequential(
                        nn.Linear(d_model, d_ff),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(d_ff, d_model),
                        nn.Dropout(dropout),
                    )

                def forward(self_inner, x, mask=None):
                    h, alpha = self_inner.attn(self_inner.ln1(x), mask)
                    x = x + h
                    x = x + self_inner.ff(self_inner.ln2(x))
                    return x, alpha

            return _StdBlock()

        def forward(self, tokens):
            """Standard forward — returns logits (compatible with eval)."""
            B, T = tokens.shape
            mask = torch.triu(
                torch.ones(T, T, device=tokens.device), diagonal=1
            ).bool()

            tok_emb = self.token_embed(tokens)
            positions = torch.arange(
                T, device=tokens.device).unsqueeze(0).expand(B, -1)
            pos_emb = self.pos_embed(positions)
            x = tok_emb + pos_emb

            if self.use_gate:
                for layer in self.layers:
                    x, _, _ = layer(x, pos_emb, mask)
            else:
                for layer in self.layers:
                    x, _ = layer(x, mask)

            hiddens = self.ln_final(x)
            logits = self.output_proj(hiddens)

            if self.use_temp and self.temp_head is not None:
                temp = self.temp_head(hiddens)  # (B, T)
                temp = temp.unsqueeze(-1)  # (B, T, 1)
                logits = logits / temp.clamp(min=0.01)

            return logits

        def forward_with_diagnostics(self, tokens):
            """Returns (logits, diagnostics_dict).

            diagnostics_dict contains:
              'temperature': (B, T) learned temperature (if use_temp)
              'gate_values': list of (B, T, n_heads) per layer (if use_gate)
            """
            B, T = tokens.shape
            mask = torch.triu(
                torch.ones(T, T, device=tokens.device), diagonal=1
            ).bool()

            tok_emb = self.token_embed(tokens)
            positions = torch.arange(
                T, device=tokens.device).unsqueeze(0).expand(B, -1)
            pos_emb = self.pos_embed(positions)
            x = tok_emb + pos_emb

            all_gates = []
            if self.use_gate:
                for layer in self.layers:
                    x, _, gate = layer(x, pos_emb, mask)
                    all_gates.append(gate.detach())
            else:
                for layer in self.layers:
                    x, _ = layer(x, mask)

            hiddens = self.ln_final(x)
            logits = self.output_proj(hiddens)

            diag = {}
            if self.use_temp and self.temp_head is not None:
                temp = self.temp_head(hiddens)  # (B, T)
                diag['temperature'] = temp.detach()
                logits = logits / temp.unsqueeze(-1).clamp(min=0.01)
            if self.use_gate:
                diag['gate_values'] = all_gates

            return logits, diag

    return RecurrenceTransformerEndogenous


# ============================================================================
# New distillation loss functions
# ============================================================================

def compute_reverse_kl_distill(student_logits, teacher_logits, unrewarded_mask,
                                n_tokens, temperature=2.0):
    """Reverse KL distillation: D_KL(P_student || P_teacher).

    Mode-seeking: student concentrates on whichever modes it finds,
    potentially ignoring low-probability regions of teacher distribution.
    """
    _ensure_torch()
    _sync_torch()
    if not unrewarded_mask.any():
        return torch.tensor(0.0, device=student_logits.device)

    s = student_logits[:, :, :n_tokens][unrewarded_mask] / temperature
    t = teacher_logits[:, :, :n_tokens][unrewarded_mask] / temperature

    s_probs = F.softmax(s, dim=-1)
    s_log_probs = F.log_softmax(s, dim=-1)
    t_log_probs = F.log_softmax(t, dim=-1)

    # D_KL(student || teacher) = sum(s_probs * (s_log_probs - t_log_probs))
    kl = (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1).mean()
    return (temperature ** 2) * kl


def compute_js_distill(student_logits, teacher_logits, unrewarded_mask,
                        n_tokens, temperature=2.0):
    """Jensen-Shannon distillation: JSD(P_student, P_teacher).

    Symmetric: JSD = 0.5 * D_KL(P||M) + 0.5 * D_KL(Q||M), M = 0.5*(P+Q).
    """
    _ensure_torch()
    _sync_torch()
    if not unrewarded_mask.any():
        return torch.tensor(0.0, device=student_logits.device)

    s = student_logits[:, :, :n_tokens][unrewarded_mask] / temperature
    t = teacher_logits[:, :, :n_tokens][unrewarded_mask] / temperature

    s_probs = F.softmax(s, dim=-1)
    t_probs = F.softmax(t, dim=-1)

    # Midpoint distribution
    m_probs = 0.5 * (s_probs + t_probs)
    # Clamp for numerical stability
    m_log_probs = torch.log(m_probs.clamp(min=1e-10))

    s_log_probs = F.log_softmax(s, dim=-1)
    t_log_probs = F.log_softmax(t, dim=-1)

    kl_s = (s_probs * (s_log_probs - m_log_probs)).sum(dim=-1).mean()
    kl_t = (t_probs * (t_log_probs - m_log_probs)).sum(dim=-1).mean()

    return (temperature ** 2) * 0.5 * (kl_s + kl_t)


# ============================================================================
# Per-position calibration metrics
# ============================================================================

def _generate_one_eval_sample(args_tuple):
    """Generate one evaluation sample (for multiprocessing)."""
    p, pi, seq_len, generator, direction = args_tuple

    if generator == 'linear':
        cfg = RecurrenceConfig(p=p, pi=pi, seq_len=seq_len)
        if direction == 'forward':
            tokens, gt, metadata = generate_recurrence_sequence(cfg)
        else:
            tokens_fwd, gt_fwd, metadata = generate_recurrence_sequence(cfg)
            tokens = list(reversed(tokens_fwd))
            gt = []
            for t in range(seq_len):
                prefix = tokens[:t]
                pred_dist = bayesian_predictive_linear_backward(prefix, p, pi)
                H = _predictive_entropy(pred_dist)
                gt.append({
                    't': t, 'entropy': H, 'pred_dist': pred_dist,
                })
    else:  # quadratic
        cfg = QuadraticConfig(p=p, pi=pi, seq_len=seq_len)
        tokens, gt, metadata = generate_quadratic_sequence(
            cfg, direction=direction)

    return tokens, gt


def evaluate_with_calibration(model, p, pi, seq_len, n_eval=2000,
                               device='cpu', generator='linear',
                               direction='forward'):
    """Evaluate model with full calibration metrics (batched).

    Beyond standard MAE, computes:
    - D_KL(P_bayes || P_model) at each position
    - Mode concentration (max probability) at each position

    Returns (metrics, per_pos) where per_pos[t] includes:
        H_model_mean, H_bayes_mean, mae_mean,
        dkl_mean, mode_conc_model_mean, mode_conc_bayes_mean
    """
    _ensure_torch()
    _sync_torch()
    from multiprocessing import Pool

    model.eval()
    eval_batch_size = 64

    # --- Phase 1: generate all samples (CPU, parallel) ---
    gen_args = [(p, pi, seq_len, generator, direction)] * n_eval
    try:
        with Pool(4) as pool:
            samples = pool.map(_generate_one_eval_sample, gen_args)
    except Exception:
        # Fallback to sequential if multiprocessing fails
        samples = [_generate_one_eval_sample(a) for a in gen_args]

    all_tokens = [s[0] for s in samples]
    all_gt = [s[1] for s in samples]

    # --- Phase 2: batched model inference ---
    all_probs = []
    for batch_start in range(0, n_eval, eval_batch_size):
        batch_end = min(batch_start + eval_batch_size, n_eval)
        batch_tokens = all_tokens[batch_start:batch_end]
        tokens_tensor = torch.tensor(batch_tokens, dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(tokens_tensor)
        probs = F.softmax(logits[:, :, :p], dim=-1).cpu().numpy()
        all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)  # (n_eval, seq_len, p)

    # --- Phase 3: compute metrics (vectorized where possible) ---
    per_position = {}

    for i in range(n_eval):
        gt = all_gt[i]
        probs_model = all_probs[i]  # (seq_len, p)

        for entry in gt:
            t = entry['t']
            if t < 1 or t >= seq_len:
                continue

            model_pos = t - 1
            if model_pos >= probs_model.shape[0]:
                continue

            model_dist = probs_model[model_pos]
            bayes_dist = entry['pred_dist']

            # Entropy (vectorized)
            md_clamped = np.maximum(model_dist, 1e-10)
            H_model = float(-np.sum(model_dist * np.log2(md_clamped)))
            H_bayes = entry['entropy']

            # MAE
            mae = abs(H_model - H_bayes)

            # D_KL(P_bayes || P_model)
            dkl = 0.0
            for v in range(p):
                pb = bayes_dist.get(v, 0.0)
                if pb > 0:
                    dkl += pb * math.log2(pb / md_clamped[v])

            # Mode concentration
            mode_conc_model = float(np.max(model_dist))
            mode_conc_bayes = max(bayes_dist.values())

            if t not in per_position:
                per_position[t] = {
                    'H_model': [], 'H_bayes': [], 'mae': [],
                    'dkl': [], 'mode_conc_model': [], 'mode_conc_bayes': [],
                }
            per_position[t]['H_model'].append(H_model)
            per_position[t]['H_bayes'].append(H_bayes)
            per_position[t]['mae'].append(mae)
            per_position[t]['dkl'].append(dkl)
            per_position[t]['mode_conc_model'].append(mode_conc_model)
            per_position[t]['mode_conc_bayes'].append(mode_conc_bayes)

    # Aggregate
    per_pos_summary = {}
    all_maes = []
    for t in sorted(per_position.keys()):
        d = per_position[t]
        per_pos_summary[t] = {
            'H_model_mean': float(np.mean(d['H_model'])),
            'H_bayes_mean': float(np.mean(d['H_bayes'])),
            'mae_mean': float(np.mean(d['mae'])),
            'dkl_mean': float(np.mean(d['dkl'])),
            'mode_conc_model_mean': float(np.mean(d['mode_conc_model'])),
            'mode_conc_bayes_mean': float(np.mean(d['mode_conc_bayes'])),
            'count': len(d['mae']),
        }
        all_maes.extend(d['mae'])

    metrics = {
        'mae_bits': float(np.mean(all_maes)) if all_maes else 0.0,
        'mae_std': float(np.std(all_maes)) if all_maes else 0.0,
    }

    return metrics, per_pos_summary


# ============================================================================
# Batch generation for quadratic/backward sequences
# ============================================================================

def _generate_one_training_sample(args_tuple):
    """Generate one training sample (for multiprocessing)."""
    p, pi, seq_len, generator, direction = args_tuple

    if generator == 'linear':
        cfg = RecurrenceConfig(p=p, pi=pi, seq_len=seq_len)
        if direction == 'forward':
            tokens, gt, metadata = generate_recurrence_sequence(cfg)
        else:
            tokens_fwd, gt_fwd, meta = generate_recurrence_sequence(cfg)
            tokens = list(reversed(tokens_fwd))
            gt = []
            for t in range(seq_len):
                prefix = tokens[:t]
                pred_dist = bayesian_predictive_linear_backward(prefix, p, pi)
                H = _predictive_entropy(pred_dist)
                gt.append({'t': t, 'entropy': H})
            metadata = meta
    else:  # quadratic
        cfg = QuadraticConfig(p=p, pi=pi, seq_len=seq_len)
        tokens, gt, metadata = generate_quadratic_sequence(
            cfg, direction=direction)

    entropies = [0.0] * seq_len
    for entry in gt:
        t = entry['t']
        if 0 <= t < seq_len:
            entropies[t] = entry['entropy']

    is_prog = 1.0 if metadata['true_class'] == 'program' else 0.0
    return tokens, entropies, is_prog


# Persistent worker pool (created on first use)
_worker_pool = None


def _get_worker_pool():
    global _worker_pool
    if _worker_pool is None:
        from multiprocessing import Pool
        _worker_pool = Pool(4)
    return _worker_pool


def generate_batch_extended(p, pi, seq_len, batch_size, device,
                            generator='linear', direction='forward'):
    """Generate batch of sequences, supporting quadratic and backward modes.

    Uses a persistent multiprocessing pool for parallel data generation.

    Returns:
        x: (B, seq_len) long tensor of tokens
        entropy_targets: (B, seq_len) float tensor of Bayesian entropy
        is_program: (B,) float tensor
    """
    _ensure_torch()
    _sync_torch()

    gen_args = [(p, pi, seq_len, generator, direction)] * batch_size

    try:
        pool = _get_worker_pool()
        results = pool.map(_generate_one_training_sample, gen_args)
    except Exception:
        results = [_generate_one_training_sample(a) for a in gen_args]

    all_tokens = [r[0] for r in results]
    all_entropies = [r[1] for r in results]
    all_is_program = [r[2] for r in results]

    x = torch.tensor(all_tokens, dtype=torch.long).to(device)
    entropy_targets = torch.tensor(
        all_entropies, dtype=torch.float32).to(device)
    is_program = torch.tensor(
        all_is_program, dtype=torch.float32).to(device)

    return x, entropy_targets, is_program


# ============================================================================
# Training loop
# ============================================================================

def train_roof(args):
    """Training loop for both experiments.

    Experiment 1 (distill_direction): trains with distillation using
    forward KL, reverse KL, or Jensen-Shannon divergence.

    Experiment 2 (causal_direction): trains on linear or quadratic
    sequences in forward or backward direction.
    """
    _ensure_torch()
    _sync_torch()

    device = _resolve_device(args.device)
    p = args.p
    vocab_size = p
    n_tokens = p
    loss_horizon = args.loss_horizon
    train_seq_len = args.train_seq_len

    endogenous_type = getattr(args, 'endogenous', 'none')
    is_endogenous = (args.experiment == 'endogenous_roof'
                     and endogenous_type != 'none')

    if is_endogenous:
        EndogenousModel = _build_endogenous_model_class(endogenous_type)
        model = EndogenousModel(
            vocab_size=vocab_size,
            n_tokens=n_tokens,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            dropout=args.dropout,
        ).to(device)
    else:
        RecurrenceTransformerSubsidy = _build_model_class()
        model = RecurrenceTransformerSubsidy(
            vocab_size=vocab_size,
            n_tokens=n_tokens,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            dropout=args.dropout,
            aux_classifier=False,
        ).to(device)

    param_count = sum(pr.numel() for pr in model.parameters())
    print(f"Model: {param_count:,} parameters on {device}")
    print(f"Experiment: {args.experiment}")
    print(f"Generator: {args.generator}, Direction: {args.prediction_direction}")
    if args.experiment == 'distill_direction':
        print(f"Distillation: {args.distill_direction}, "
              f"lambda={args.subsidy_lambda}")
    if is_endogenous:
        print(f"Endogenous mechanism: {endogenous_type}")
    print(f"Loss horizon: 1-{loss_horizon} (of {train_seq_len})")

    # Load teacher for distillation experiment
    teacher = None
    if args.experiment == 'distill_direction' and args.subsidy_lambda > 0:
        teacher = RecurrenceTransformerSubsidy(
            vocab_size=vocab_size, n_tokens=n_tokens,
            d_model=args.d_model, n_layers=args.n_layers,
            n_heads=args.n_heads, d_ff=args.d_ff,
            dropout=0.0, aux_classifier=False,
        ).to(device)

        if args.control:
            print("  Teacher: RANDOM (untrained) — control condition")
        else:
            if not args.teacher_checkpoint:
                raise ValueError(
                    "Distillation requires --teacher_checkpoint "
                    "(or --control for random teacher)")
            state = torch.load(args.teacher_checkpoint, map_location=device,
                               weights_only=True)
            teacher.load_state_dict(state)
            print(f"  Teacher loaded from {args.teacher_checkpoint}")

        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_steps)

    os.makedirs(args.output_dir, exist_ok=True)
    best_mae = float('inf')
    losses_ce = []
    losses_sub = []

    # Pre-compute masks
    rewarded_mask = torch.zeros(
        1, train_seq_len, dtype=torch.bool, device=device)

    # Support non-contiguous loss positions (e.g., "1-5,13-15")
    loss_positions_str = getattr(args, 'loss_positions', None)
    if loss_positions_str:
        for rng in loss_positions_str.split(','):
            rng = rng.strip()
            if '-' in rng:
                lo, hi = rng.split('-')
                # loss_positions "1-5" means positions 1..5 in 1-indexed
                # which maps to mask positions 0..4 (0-indexed prediction positions)
                for pos in range(int(lo) - 1, int(hi)):
                    if 0 <= pos < train_seq_len:
                        rewarded_mask[0, pos] = True
            else:
                pos = int(rng) - 1
                if 0 <= pos < train_seq_len:
                    rewarded_mask[0, pos] = True
    else:
        rewarded_mask[0, :loss_horizon] = True

    unrewarded_mask = torch.zeros(
        1, train_seq_len, dtype=torch.bool, device=device)
    unrewarded_mask[0, loss_horizon:train_seq_len - 1] = True

    # Entropy subsidy mask (for calibration experiment)
    entropy_mask_type = getattr(args, 'entropy_mask', 'all')
    entropy_subsidy_mask = torch.zeros(
        1, train_seq_len, dtype=torch.bool, device=device)
    if entropy_mask_type == 'all':
        entropy_subsidy_mask[0, loss_horizon:train_seq_len - 1] = True
    elif entropy_mask_type == 'single_6':
        if 5 < train_seq_len - 1:
            entropy_subsidy_mask[0, 5] = True  # position 5 predicts token 6
    elif entropy_mask_type == 'single_10':
        if 9 < train_seq_len - 1:
            entropy_subsidy_mask[0, 9] = True
    elif entropy_mask_type == 'single_14':
        if 13 < train_seq_len - 1:
            entropy_subsidy_mask[0, 13] = True
    elif entropy_mask_type == 'every_other':
        for pos in [5, 7, 9, 11, 13]:  # positions predicting tokens 6,8,10,12,14
            if pos < train_seq_len - 1:
                entropy_subsidy_mask[0, pos] = True

    for step in range(1, args.n_steps + 1):
        model.train()

        x, entropy_targets, is_program = generate_batch_extended(
            p, args.pi, train_seq_len, args.batch_size, device,
            generator=args.generator,
            direction=args.prediction_direction,
        )
        B = x.shape[0]

        rew_mask = rewarded_mask.expand(B, -1)
        unrew_mask = unrewarded_mask.expand(B, -1)

        targets = x[:, 1:]
        logits = model(x)
        pred_logits = logits[:, :-1, :n_tokens]

        # CE loss at rewarded positions
        ce_mask = rew_mask[:, :-1]
        ce_loss = _masked_ce_loss(pred_logits, targets, ce_mask, n_tokens)

        # Subsidy loss
        sub_loss = torch.tensor(0.0, device=device)

        # Distillation experiment
        if (args.experiment == 'distill_direction'
                and teacher is not None
                and args.subsidy_lambda > 0):
            with torch.no_grad():
                teacher_logits = teacher(x)
            unrew_shifted = unrew_mask[:, :-1]

            if args.distill_direction == 'forward':
                sub_loss = compute_distill_subsidy(
                    logits[:, :-1, :], teacher_logits[:, :-1, :],
                    unrew_shifted, n_tokens, temperature=2.0)
            elif args.distill_direction == 'reverse':
                sub_loss = compute_reverse_kl_distill(
                    logits[:, :-1, :], teacher_logits[:, :-1, :],
                    unrew_shifted, n_tokens, temperature=2.0)
            elif args.distill_direction == 'js':
                sub_loss = compute_js_distill(
                    logits[:, :-1, :], teacher_logits[:, :-1, :],
                    unrew_shifted, n_tokens, temperature=2.0)

        # Calibration experiment — entropy subsidy with configurable targets and mask
        if args.experiment == 'calibration' and args.subsidy_lambda > 0:
            entropy_start = getattr(args, 'entropy_start_step', 0)
            entropy_end = getattr(args, 'entropy_end_step', 999999)
            if entropy_start <= step <= entropy_end:
                ent_mask_shifted = entropy_subsidy_mask.expand(B, -1)[:, :-1]
                ent_targets_shifted = entropy_targets[:, 1:]

                # Apply target mode
                target_mode = getattr(args, 'entropy_target_mode', 'correct')
                if target_mode == 'correct':
                    pass  # Use Bayesian targets as-is
                elif target_mode == 'constant':
                    c = getattr(args, 'entropy_constant', 2.08)
                    ent_targets_shifted = torch.full_like(
                        ent_targets_shifted, c)
                elif target_mode == 'random':
                    ent_targets_shifted = (
                        torch.rand_like(ent_targets_shifted) * math.log2(p))

                sub_loss = compute_entropy_subsidy(
                    logits[:, :-1, :], ent_targets_shifted,
                    ent_mask_shifted, n_tokens)

        total_loss = ce_loss + args.subsidy_lambda * sub_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses_ce.append(ce_loss.item())
        losses_sub.append(sub_loss.item())

        if step % args.log_every == 0:
            recent_ce = np.mean(losses_ce[-args.log_every:])
            recent_sub = np.mean(losses_sub[-args.log_every:])
            print(f"  Step {step}/{args.n_steps}: CE={recent_ce:.4f}, "
                  f"sub={recent_sub:.4f}", flush=True)

        if step % args.eval_every == 0:
            metrics, per_pos = evaluate_with_calibration(
                model, p, args.pi, train_seq_len,
                n_eval=200, device=str(device),
                generator=args.generator,
                direction=args.prediction_direction,
            )
            wm = compute_wall_metrics(per_pos, loss_horizon, train_seq_len)

            print(f"  Eval: MAE={metrics['mae_bits']:.4f}, "
                  f"WR={wm['wall_ratio']:.1f}x, "
                  f"trained={wm['trained_mae']:.4f}, "
                  f"untrained={wm['untrained_mae']:.4f}", flush=True)

            for t in sorted(per_pos.keys()):
                pp = per_pos[t]
                marker = " [T]" if t <= loss_horizon else " [U]"
                print(f"    t={t:2d}: MAE={pp['mae_mean']:.4f} "
                      f"DKL={pp['dkl_mean']:.4f} "
                      f"MC={pp['mode_conc_model_mean']:.3f}{marker}")

            # Log endogenous diagnostics
            if is_endogenous and hasattr(model, 'forward_with_diagnostics'):
                with torch.no_grad():
                    sample_x, _, _ = generate_batch_extended(
                        p, args.pi, train_seq_len, 32, device,
                        generator=args.generator,
                        direction=args.prediction_direction)
                    _, diag = model.forward_with_diagnostics(sample_x)
                    if 'temperature' in diag:
                        temp = diag['temperature'].mean(dim=0)  # (T,)
                        temp_str = " ".join(
                            f"{temp[t].item():.2f}" for t in range(
                                min(train_seq_len, len(temp))))
                        print(f"    Temp: [{temp_str}]")
                    if 'gate_values' in diag:
                        # Mean gate across batch and heads for last layer
                        last_gate = diag['gate_values'][-1].mean(
                            dim=(0, 2))  # (T,)
                        gate_str = " ".join(
                            f"{last_gate[t].item():.2f}" for t in range(
                                min(train_seq_len, len(last_gate))))
                        print(f"    Gate (last layer): [{gate_str}]")

            if metrics['mae_bits'] < best_mae:
                best_mae = metrics['mae_bits']
                torch.save(
                    model.state_dict(),
                    os.path.join(args.output_dir, 'best_model.pt'))

            # Periodic checkpoints for degradation tracking (Experiment 5)
            checkpoint_every = getattr(args, 'checkpoint_every', 0)
            if checkpoint_every > 0 and step % checkpoint_every == 0:
                ckpt_path = os.path.join(
                    args.output_dir, f'checkpoint_step{step}.pt')
                torch.save(model.state_dict(), ckpt_path)
                # Save per-position metrics at this checkpoint
                ckpt_metrics_path = os.path.join(
                    args.output_dir, f'metrics_step{step}.json')
                with open(ckpt_metrics_path, 'w') as f:
                    json.dump({
                        'step': step,
                        'metrics': metrics,
                        'per_position': {str(k): v for k, v in per_pos.items()},
                        'wall_metrics': wm,
                    }, f, indent=2, default=float)

    # ====================================================================
    # Final evaluation
    # ====================================================================
    print(f"\n{'='*70}")
    print("FINAL EVALUATION")
    print(f"{'='*70}")

    metrics, per_pos = evaluate_with_calibration(
        model, p, args.pi, train_seq_len,
        n_eval=2000, device=str(device),
        generator=args.generator,
        direction=args.prediction_direction,
    )
    wm = compute_wall_metrics(per_pos, loss_horizon, train_seq_len)
    ef = compute_erosion_fraction(wm['wall_ratio'])

    print(f"\n  Experiment: {args.experiment}")
    print(f"  Generator: {args.generator}, "
          f"Direction: {args.prediction_direction}")
    if args.experiment == 'distill_direction':
        print(f"  Distill direction: {args.distill_direction} "
              f"({'control' if args.control else 'active'})")
        print(f"  Lambda: {args.subsidy_lambda}")
    print(f"  Loss horizon: 1-{loss_horizon}")
    print()

    for t in sorted(per_pos.keys()):
        pp = per_pos[t]
        marker = " [TRAINED]" if t <= loss_horizon else " [UNTRAINED]"
        print(f"    t={t:2d}: H_model={pp['H_model_mean']:.4f}, "
              f"H_bayes={pp['H_bayes_mean']:.4f}, "
              f"MAE={pp['mae_mean']:.4f}, "
              f"DKL={pp['dkl_mean']:.4f}, "
              f"MC_m={pp['mode_conc_model_mean']:.3f}, "
              f"MC_b={pp['mode_conc_bayes_mean']:.3f}{marker}")

    print(f"\n  Trained MAE  (1-{loss_horizon}): {wm['trained_mae']:.4f} bits")
    print(f"  Untrained MAE ({loss_horizon+1}-{train_seq_len-1}): "
          f"{wm['untrained_mae']:.4f} bits")
    print(f"  Wall Ratio: {wm['wall_ratio']:.2f}x")
    print(f"  Erosion Fraction: {ef:.4f}")

    # Endogenous diagnostics for final results
    endogenous_diag = {}
    if is_endogenous and hasattr(model, 'forward_with_diagnostics'):
        with torch.no_grad():
            sample_x, _, _ = generate_batch_extended(
                p, args.pi, train_seq_len, 200, device,
                generator=args.generator,
                direction=args.prediction_direction)
            _, diag = model.forward_with_diagnostics(sample_x)
            if 'temperature' in diag:
                temp = diag['temperature'].mean(dim=0).cpu().tolist()
                endogenous_diag['temperature_per_position'] = {
                    str(t): temp[t] for t in range(len(temp))}
                print(f"\n  Learned temperature per position:")
                for t in range(len(temp)):
                    marker = " [T]" if t < loss_horizon else " [U]"
                    print(f"    t={t:2d}: T={temp[t]:.4f}{marker}")
            if 'gate_values' in diag:
                # Per-layer, per-position mean gate (across batch and heads)
                gate_per_layer = {}
                for layer_idx, gv in enumerate(diag['gate_values']):
                    g = gv.mean(dim=(0, 2)).cpu().tolist()  # (T,)
                    gate_per_layer[str(layer_idx)] = {
                        str(t): g[t] for t in range(len(g))}
                endogenous_diag['gate_per_layer_position'] = gate_per_layer
                # Print last layer
                last = diag['gate_values'][-1].mean(dim=(0, 2)).cpu()
                print(f"\n  Gate values (last layer) per position:")
                for t in range(len(last)):
                    marker = " [T]" if t < loss_horizon else " [U]"
                    print(f"    t={t:2d}: gate={last[t]:.4f}{marker}")

    results = {
        'experiment': args.experiment,
        'generator': args.generator,
        'prediction_direction': args.prediction_direction,
        'distill_direction': getattr(args, 'distill_direction', None),
        'control': getattr(args, 'control', False),
        'subsidy_lambda': getattr(args, 'subsidy_lambda', 0.0),
        'endogenous': endogenous_type,
        'loss_horizon': loss_horizon,
        'train_seq_len': train_seq_len,
        'metrics': metrics,
        'per_position': {str(k): v for k, v in per_pos.items()},
        'wall_metrics': wm,
        'erosion_fraction': ef,
        'endogenous_diagnostics': endogenous_diag,
    }

    results_path = os.path.join(args.output_dir, 'roof_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to {results_path}", flush=True)

    return results


# ============================================================================
# Full experimental matrices
# ============================================================================

def run_distill_direction_matrix(args):
    """Run Experiment 1: all distillation directions x lambdas x seeds."""
    from wall_erosion_experiment import train_teacher

    base_output = args.output_dir
    all_results = []

    conditions = [
        # (direction, control, lambdas)
        ('forward', False, [0.1, 0.5, 1.0]),
        ('forward', True, [0.5]),           # random teacher control
        ('reverse', False, [0.1, 0.5, 1.0]),
        ('reverse', True, [0.5]),
        ('js', False, [0.1, 0.5, 1.0]),
        ('js', True, [0.5]),
    ]

    for seed in args.seeds:
        print(f"\n{'#'*70}")
        print(f"# SEED {seed}")
        print(f"{'#'*70}")

        _seed_all(seed)

        # Train teacher for this seed
        teacher_dir = os.path.join(base_output, f'teacher_seed{seed}')
        teacher_args = argparse.Namespace(**vars(args))
        teacher_args.output_dir = teacher_dir
        teacher_ckpt = train_teacher(teacher_args)

        # Baseline (no distillation)
        _seed_all(seed)
        baseline_args = argparse.Namespace(**vars(args))
        baseline_args.subsidy_lambda = 0.0
        baseline_args.distill_direction = 'forward'
        baseline_args.control = False
        baseline_args.teacher_checkpoint = None
        baseline_args.output_dir = os.path.join(
            base_output, f'baseline_seed{seed}')
        result = train_roof(baseline_args)
        result['condition'] = 'baseline'
        result['seed'] = seed
        all_results.append(result)

        for direction, control, lambdas in conditions:
            for lam in lambdas:
                ctrl_str = "ctrl" if control else "active"
                cond_name = (f"distill_{direction}_{ctrl_str}_"
                             f"lam{lam}_seed{seed}")
                cond_dir = os.path.join(base_output, cond_name)

                print(f"\n{'='*70}")
                print(f"CONDITION: {cond_name}")
                print(f"{'='*70}")

                _seed_all(seed)

                cond_args = argparse.Namespace(**vars(args))
                cond_args.distill_direction = direction
                cond_args.control = control
                cond_args.subsidy_lambda = lam
                cond_args.output_dir = cond_dir
                cond_args.teacher_checkpoint = teacher_ckpt

                result = train_roof(cond_args)
                result['condition'] = cond_name
                result['seed'] = seed
                all_results.append(result)

    # Save summary
    summary_path = os.path.join(base_output, 'distill_direction_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nSummary saved to {summary_path}")

    _print_summary_table(all_results)


def run_causal_direction_matrix(args):
    """Run Experiment 2: all generator x direction x horizon x seeds."""
    base_output = args.output_dir
    all_results = []

    conditions = [
        # (generator, direction, loss_horizon)
        ('linear', 'forward', 5),
        ('linear', 'backward', 5),
        ('quadratic', 'forward', 5),
        ('quadratic', 'backward', 5),
        ('quadratic', 'forward', 15),   # full horizon
        ('quadratic', 'backward', 15),  # full horizon
    ]

    for seed in args.seeds:
        print(f"\n{'#'*70}")
        print(f"# SEED {seed}")
        print(f"{'#'*70}")

        for generator, direction, horizon in conditions:
            cond_name = (f"{generator}_{direction}_K{horizon}_seed{seed}")
            cond_dir = os.path.join(base_output, cond_name)

            print(f"\n{'='*70}")
            print(f"CONDITION: {cond_name}")
            print(f"{'='*70}")

            _seed_all(seed)

            cond_args = argparse.Namespace(**vars(args))
            cond_args.generator = generator
            cond_args.prediction_direction = direction
            cond_args.loss_horizon = horizon
            cond_args.subsidy_lambda = 0.0
            cond_args.control = False
            cond_args.output_dir = cond_dir

            result = train_roof(cond_args)
            result['condition'] = cond_name
            result['seed'] = seed
            all_results.append(result)

    # Save summary
    summary_path = os.path.join(base_output, 'causal_direction_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nSummary saved to {summary_path}")

    _print_summary_table(all_results)


def run_endogenous_matrix(args):
    """Run Experiment 3: all endogenous mechanisms x seeds."""
    base_output = args.output_dir
    all_results = []

    conditions = [
        # (endogenous_type,)
        ('none',),         # baseline (no endogenous mechanism)
        ('temperature',),  # learned temperature head
        ('gate',),         # positional forget-gate
        ('both',),         # temperature + gate combined
    ]

    for seed in args.seeds:
        print(f"\n{'#'*70}")
        print(f"# SEED {seed}")
        print(f"{'#'*70}")

        for (endo_type,) in conditions:
            cond_name = f"endogenous_{endo_type}_seed{seed}"
            cond_dir = os.path.join(base_output, cond_name)

            print(f"\n{'='*70}")
            print(f"CONDITION: {cond_name}")
            print(f"{'='*70}")

            _seed_all(seed)

            cond_args = argparse.Namespace(**vars(args))
            cond_args.endogenous = endo_type
            cond_args.subsidy_lambda = 0.0
            cond_args.control = False
            cond_args.output_dir = cond_dir

            result = train_roof(cond_args)
            result['condition'] = cond_name
            result['seed'] = seed
            all_results.append(result)

    # Save summary
    summary_path = os.path.join(base_output, 'endogenous_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nSummary saved to {summary_path}")

    _print_summary_table(all_results)


def run_calibration_matrix(args):
    """Run calibration barrier experiments (3-5) for all conditions."""
    base_output = args.output_dir
    all_results = []

    LAMBDA = 0.1  # Fixed across all calibration experiments

    # --- Experiment 3: Wrong entropy targets ---
    exp3_conditions = [
        # (target_mode, entropy_constant, label)
        ('correct', None, 'ent_correct'),
        ('constant', 2.08, 'ent_const_2.08'),
        ('constant', 1.0, 'ent_const_1.0'),
        ('constant', 3.0, 'ent_const_3.0'),
        ('constant', 0.5, 'ent_const_0.5'),
        ('constant', math.log2(17), 'ent_uniform'),
        ('random', None, 'ent_random'),
    ]

    # --- Experiment 4: Propagation ---
    exp4_conditions = [
        # (entropy_mask, label)
        ('single_6', 'prop_single6'),
        ('single_10', 'prop_single10'),
        ('single_14', 'prop_single14'),
        ('every_other', 'prop_every_other'),
        ('all', 'prop_all'),
    ]

    # --- Experiment 5: Timing ---
    exp5_conditions = [
        # (start_step, end_step, checkpoint_every, label)
        (0, 999999, 0, 'timing_from_start'),
        (50000, 999999, 0, 'timing_late_50k'),
        (100000, 999999, 0, 'timing_late_100k'),
        (0, 50000, 10000, 'timing_remove_50k'),
        (50000, 60000, 10000, 'timing_pulse_50k_60k'),
    ]

    for seed in args.seeds:
        print(f"\n{'#'*70}")
        print(f"# SEED {seed}")
        print(f"{'#'*70}")

        # Experiment 3
        for target_mode, const_val, label in exp3_conditions:
            cond_name = f"{label}_seed{seed}"
            print(f"\n{'='*70}")
            print(f"CONDITION: {cond_name}")
            print(f"{'='*70}")

            _seed_all(seed)
            cond_args = argparse.Namespace(**vars(args))
            cond_args.experiment = 'calibration'
            cond_args.subsidy_lambda = LAMBDA
            cond_args.entropy_target_mode = target_mode
            cond_args.entropy_constant = const_val if const_val else 2.08
            cond_args.entropy_mask = 'all'
            cond_args.entropy_start_step = 0
            cond_args.entropy_end_step = 999999
            cond_args.checkpoint_every = 0
            cond_args.output_dir = os.path.join(base_output, cond_name)

            result = train_roof(cond_args)
            result['condition'] = cond_name
            result['seed'] = seed
            all_results.append(result)

        # Experiment 4
        for mask_type, label in exp4_conditions:
            cond_name = f"{label}_seed{seed}"
            print(f"\n{'='*70}")
            print(f"CONDITION: {cond_name}")
            print(f"{'='*70}")

            _seed_all(seed)
            cond_args = argparse.Namespace(**vars(args))
            cond_args.experiment = 'calibration'
            cond_args.subsidy_lambda = LAMBDA
            cond_args.entropy_target_mode = 'correct'
            cond_args.entropy_constant = 2.08
            cond_args.entropy_mask = mask_type
            cond_args.entropy_start_step = 0
            cond_args.entropy_end_step = 999999
            cond_args.checkpoint_every = 0
            cond_args.output_dir = os.path.join(base_output, cond_name)

            result = train_roof(cond_args)
            result['condition'] = cond_name
            result['seed'] = seed
            all_results.append(result)

        # Experiment 5
        for start, end, ckpt_every, label in exp5_conditions:
            cond_name = f"{label}_seed{seed}"
            print(f"\n{'='*70}")
            print(f"CONDITION: {cond_name}")
            print(f"{'='*70}")

            _seed_all(seed)
            cond_args = argparse.Namespace(**vars(args))
            cond_args.experiment = 'calibration'
            cond_args.subsidy_lambda = LAMBDA
            cond_args.entropy_target_mode = 'correct'
            cond_args.entropy_constant = 2.08
            cond_args.entropy_mask = 'all'
            cond_args.entropy_start_step = start
            cond_args.entropy_end_step = end
            cond_args.checkpoint_every = ckpt_every
            cond_args.output_dir = os.path.join(base_output, cond_name)

            result = train_roof(cond_args)
            result['condition'] = cond_name
            result['seed'] = seed
            all_results.append(result)

    # Save summary
    summary_path = os.path.join(base_output, 'calibration_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nSummary saved to {summary_path}")

    _print_summary_table(all_results)


def _print_summary_table(all_results):
    """Print a summary table of results."""
    print(f"\n{'='*90}")
    print("SUMMARY TABLE")
    print(f"{'='*90}")
    print(f"{'Condition':<50} {'WR':>8} {'EF':>8} "
          f"{'Trained':>10} {'Untrained':>10}")
    print("-" * 90)
    for r in all_results:
        wm = r['wall_metrics']
        print(f"{r['condition']:<50} {wm['wall_ratio']:>8.2f} "
              f"{r['erosion_fraction']:>8.4f} "
              f"{wm['trained_mae']:>10.4f} "
              f"{wm['untrained_mae']:>10.4f}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Roof experiment — KL asymmetry and causal transfer')

    # Experiment selection
    parser.add_argument('--experiment',
                        choices=['distill_direction', 'causal_direction',
                                 'endogenous_roof', 'calibration'],
                        required=True,
                        help='Which experiment to run')
    parser.add_argument('--run_matrix', action='store_true',
                        help='Run the full experimental matrix')

    # Experiment 1: distillation direction
    parser.add_argument('--distill_direction',
                        choices=['forward', 'reverse', 'js'],
                        default='forward',
                        help='KL divergence direction for distillation')
    parser.add_argument('--subsidy_lambda', type=float, default=0.0,
                        help='Weight for distillation loss')
    parser.add_argument('--control', action='store_true',
                        help='Use random teacher (control)')
    parser.add_argument('--teacher_checkpoint', type=str, default=None,
                        help='Path to teacher model checkpoint')

    # Calibration experiment (Experiments 3-5)
    parser.add_argument('--entropy_target_mode',
                        choices=['correct', 'constant', 'random'],
                        default='correct',
                        help='How to generate entropy targets')
    parser.add_argument('--entropy_constant', type=float, default=2.08,
                        help='Constant entropy target value (for --entropy_target_mode constant)')
    parser.add_argument('--entropy_mask',
                        choices=['all', 'single_6', 'single_10', 'single_14',
                                 'every_other'],
                        default='all',
                        help='Which positions receive entropy signal')
    parser.add_argument('--entropy_start_step', type=int, default=0,
                        help='Step at which entropy subsidy begins')
    parser.add_argument('--entropy_end_step', type=int, default=999999,
                        help='Step at which entropy subsidy ends')
    parser.add_argument('--checkpoint_every', type=int, default=0,
                        help='Save checkpoints every N steps (0=disabled)')
    parser.add_argument('--loss_positions', type=str, default=None,
                        help='Non-contiguous loss positions (e.g., "1-5,13-15")')

    # Endogenous roof
    parser.add_argument('--endogenous',
                        choices=['none', 'temperature', 'gate', 'both'],
                        default='none',
                        help='Endogenous roof mechanism')

    # Experiment 2: causal direction
    parser.add_argument('--generator',
                        choices=['linear', 'quadratic'],
                        default='linear',
                        help='Data generating process')
    parser.add_argument('--prediction_direction',
                        choices=['forward', 'backward'],
                        default='forward',
                        help='Prediction direction')

    # Task parameters
    parser.add_argument('--p', type=int, default=17)
    parser.add_argument('--pi', type=float, default=0.5)
    parser.add_argument('--train_seq_len', type=int, default=16)
    parser.add_argument('--loss_horizon', type=int, default=5)

    # Architecture
    parser.add_argument('--d_model', type=int, default=192)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=6)
    parser.add_argument('--d_ff', type=int, default=768)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Training
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_steps', type=int, default=150000)
    parser.add_argument('--eval_every', type=int, default=10000)
    parser.add_argument('--log_every', type=int, default=500)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--output_dir', type=str,
                        default='results/roof')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    if args.run_matrix:
        if args.experiment == 'distill_direction':
            run_distill_direction_matrix(args)
        elif args.experiment == 'causal_direction':
            run_causal_direction_matrix(args)
        elif args.experiment == 'endogenous_roof':
            run_endogenous_matrix(args)
        elif args.experiment == 'calibration':
            run_calibration_matrix(args)
        return

    # Single condition
    for seed in args.seeds:
        print(f"\n{'='*70}")
        print(f"ROOF EXPERIMENT: {args.experiment}, seed={seed}")
        print(f"{'='*70}")

        _seed_all(seed)

        seed_dir = os.path.join(
            args.output_dir,
            f'{args.experiment}_{args.generator}_'
            f'{args.prediction_direction}_seed{seed}'
        )
        seed_args = argparse.Namespace(**vars(args))
        seed_args.output_dir = seed_dir
        train_roof(seed_args)


if __name__ == '__main__':
    main()
