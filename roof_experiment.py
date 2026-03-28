"""
Roof Experiment — Testing KL Asymmetry as the Mechanism of Causal Transfer
============================================================================

Two experiments testing the Maintaining Divergence framework's claim that
KL divergence direction determines whether causal structure transfers:

Experiment 1 — Distillation Direction:
  Forward KL (mode-covering), Reverse KL (mode-seeking), and Jensen-Shannon
  (symmetric) distillation from the same teacher. Tests whether the direction
  of knowledge transfer determines uncertainty calibration at unrewarded positions.

Experiment 2 — Non-Invertible Generating Process:
  Quadratic recurrence x_{t+1} = x_t^2 + b mod 17 (many-to-one) vs
  linear recurrence x_{t+1} = ax_t + b mod 17 (one-to-one). Forward vs
  backward prediction. Tests whether the statistical asymmetry created by
  genuine causation is detectable only in the causal direction.

Usage:
    # Experiment 1: single distillation condition
    python roof_experiment.py --experiment distill_direction \
        --distill_direction reverse --subsidy_lambda 0.5 \
        --teacher_checkpoint results/wall_erosion/teacher_seed42/best_model.pt \
        --seeds 42 --device mps

    # Experiment 1: full matrix
    python roof_experiment.py --experiment distill_direction --run_matrix \
        --seeds 42 43 44 --device mps

    # Experiment 2: single causal direction condition
    python roof_experiment.py --experiment causal_direction \
        --generator quadratic --prediction_direction forward \
        --seeds 42 --device mps

    # Experiment 2: full matrix
    python roof_experiment.py --experiment causal_direction --run_matrix \
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
    compute_distill_subsidy, compute_wall_metrics, compute_erosion_fraction,
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

def evaluate_with_calibration(model, p, pi, seq_len, n_eval=2000,
                               device='cpu', generator='linear',
                               direction='forward'):
    """Evaluate model with full calibration metrics.

    Beyond standard MAE, computes:
    - D_KL(P_bayes || P_model) at each position
    - Mode concentration (max probability) at each position

    Returns (metrics, per_pos) where per_pos[t] includes:
        H_model_mean, H_bayes_mean, mae_mean,
        dkl_mean, mode_conc_model_mean, mode_conc_bayes_mean
    """
    _ensure_torch()
    _sync_torch()

    model.eval()
    per_position = {}

    for i in range(n_eval):
        # Generate sequence and ground truth
        if generator == 'linear':
            cfg = RecurrenceConfig(p=p, pi=pi, seq_len=seq_len)
            if direction == 'forward':
                tokens, gt, metadata = generate_recurrence_sequence(cfg)
            else:
                # Reverse the sequence for backward linear prediction
                tokens_fwd, gt_fwd, metadata = generate_recurrence_sequence(cfg)
                tokens = list(reversed(tokens_fwd))
                # Recompute ground truth for reversed sequence
                gt = []
                for t in range(seq_len):
                    prefix = tokens[:t]
                    pred_dist = bayesian_predictive_linear_backward(
                        prefix, p, pi)
                    H = _predictive_entropy(pred_dist)
                    from recurrence_bwt import class_posterior_recurrence
                    w = class_posterior_recurrence(prefix, p, pi)
                    gt.append({
                        't': t,
                        'entropy': H,
                        'pred_dist': pred_dist,
                        'p_program': w,
                    })
        else:  # quadratic
            cfg = QuadraticConfig(p=p, pi=pi, seq_len=seq_len)
            tokens, gt, metadata = generate_quadratic_sequence(
                cfg, direction=direction)

        # Model forward pass
        tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(tokens_tensor)

        probs_model = F.softmax(logits[0, :, :p], dim=-1).cpu().numpy()

        for entry in gt:
            t = entry['t']
            if t < 1 or t >= seq_len:
                continue

            # Model predicts token at position t from logits at position t-1
            model_pos = t - 1
            if model_pos >= probs_model.shape[0]:
                continue

            model_dist = probs_model[model_pos]
            bayes_dist = entry['pred_dist']

            # Entropy
            H_model = -sum(
                model_dist[v] * np.log2(max(model_dist[v], 1e-10))
                for v in range(p)
            )
            H_bayes = entry['entropy']

            # MAE
            mae = abs(H_model - H_bayes)

            # D_KL(P_bayes || P_model)
            dkl = 0.0
            for v in range(p):
                pb = bayes_dist.get(v, 0.0)
                pm = max(float(model_dist[v]), 1e-10)
                if pb > 0:
                    dkl += pb * math.log2(pb / pm)

            # Mode concentration (max probability)
            mode_conc_model = float(max(model_dist))
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

def generate_batch_extended(p, pi, seq_len, batch_size, device,
                            generator='linear', direction='forward'):
    """Generate batch of sequences, supporting quadratic and backward modes.

    Returns:
        x: (B, seq_len) long tensor of tokens
        entropy_targets: (B, seq_len) float tensor of Bayesian entropy
        is_program: (B,) float tensor
    """
    _ensure_torch()
    _sync_torch()

    all_tokens = []
    all_entropies = []
    all_is_program = []

    for _ in range(batch_size):
        if generator == 'linear':
            cfg = RecurrenceConfig(p=p, pi=pi, seq_len=seq_len)
            if direction == 'forward':
                tokens, gt, metadata = generate_recurrence_sequence(cfg)
            else:
                tokens_fwd, gt_fwd, meta = generate_recurrence_sequence(cfg)
                tokens = list(reversed(tokens_fwd))
                # Recompute gt for reversed sequence
                gt = []
                for t in range(seq_len):
                    prefix = tokens[:t]
                    pred_dist = bayesian_predictive_linear_backward(
                        prefix, p, pi)
                    H = _predictive_entropy(pred_dist)
                    gt.append({'t': t, 'entropy': H})
                metadata = meta
        else:  # quadratic
            cfg = QuadraticConfig(p=p, pi=pi, seq_len=seq_len)
            tokens, gt, metadata = generate_quadratic_sequence(
                cfg, direction=direction)

        all_tokens.append(tokens)

        entropies = [0.0] * seq_len
        for entry in gt:
            t = entry['t']
            if 0 <= t < seq_len:
                entropies[t] = entry['entropy']
        all_entropies.append(entropies)

        all_is_program.append(
            1.0 if metadata['true_class'] == 'program' else 0.0)

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
    RecurrenceTransformerSubsidy = _build_model_class()

    device = _resolve_device(args.device)
    p = args.p
    vocab_size = p
    n_tokens = p
    loss_horizon = args.loss_horizon
    train_seq_len = args.train_seq_len

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
    rewarded_mask[0, :loss_horizon] = True

    unrewarded_mask = torch.zeros(
        1, train_seq_len, dtype=torch.bool, device=device)
    unrewarded_mask[0, loss_horizon:train_seq_len - 1] = True

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

        # Subsidy loss (distillation experiment only)
        sub_loss = torch.tensor(0.0, device=device)

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
                n_eval=500, device=str(device),
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

            if metrics['mae_bits'] < best_mae:
                best_mae = metrics['mae_bits']
                torch.save(
                    model.state_dict(),
                    os.path.join(args.output_dir, 'best_model.pt'))

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

    results = {
        'experiment': args.experiment,
        'generator': args.generator,
        'prediction_direction': args.prediction_direction,
        'distill_direction': getattr(args, 'distill_direction', None),
        'control': getattr(args, 'control', False),
        'subsidy_lambda': getattr(args, 'subsidy_lambda', 0.0),
        'loss_horizon': loss_horizon,
        'train_seq_len': train_seq_len,
        'metrics': metrics,
        'per_position': {str(k): v for k, v in per_pos.items()},
        'wall_metrics': wm,
        'erosion_fraction': ef,
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
                        choices=['distill_direction', 'causal_direction'],
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
