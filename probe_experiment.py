"""
Probe Experiment — Testing the Calibration Barrier Hypothesis
==============================================================

Three diagnostic modes that require no training, only a saved checkpoint:

Experiment 0 — Cosine Similarity (--mode cosine_sim):
  Per-layer cosine similarity between hidden states at trained vs untrained
  positions. Tests whether Rigollet's global clustering spans the wall.

Experiment 1 — Temperature Sweep (--mode temperature_sweep):
  Post-hoc rescaling of logits at unrewarded positions. Tests whether the
  logit direction is already correct and only the scale is wrong.

Experiment 2 — Linear Probing (--mode probe):
  Per-layer per-position linear probes for next-token identity, recurrence
  parameters, and Bayesian entropy. Tests whether the compiled circuit
  encodes the correct answer at unrewarded positions.

Usage:
    # Experiment 0: cosine similarity (seconds)
    python probe_experiment.py --mode cosine_sim \
        --checkpoint results/roof_distill/baseline_seed42/best_model.pt \
        --device cuda --n_eval 500

    # Experiment 1: temperature sweep (minutes)
    python probe_experiment.py --mode temperature_sweep \
        --checkpoint results/roof_distill/baseline_seed42/best_model.pt \
        --device cuda --n_eval 2000

    # Experiment 2: probing (minutes per layer)
    python probe_experiment.py --mode probe \
        --checkpoint results/roof_distill/baseline_seed42/best_model.pt \
        --device cuda --n_eval 5000 --probe_steps 2000
"""

import os
import math
import argparse
import json
import numpy as np

from recurrence_bwt import (
    RecurrenceConfig,
    generate_recurrence_sequence,
    bayesian_predictive_recurrence,
    class_posterior_recurrence,
    count_consistent_recurrences,
    _predictive_entropy,
)
from wall_erosion_experiment import (
    _ensure_torch, _resolve_device, _seed_all, _build_model_class,
)

# Lazy torch
torch = None
nn = None
F = None


def _sync_torch():
    global torch, nn, F
    _ensure_torch()
    import torch as _t, torch.nn as _nn, torch.nn.functional as _F
    torch, nn, F = _t, _nn, _F


# ============================================================================
# Per-layer hidden state extraction
# ============================================================================

def extract_per_layer_hiddens(model, tokens):
    """Run model layer by layer, collecting hidden states at each layer.

    Args:
        model: RecurrenceTransformerSubsidy instance
        tokens: (B, T) long tensor

    Returns:
        layer_hiddens: list of (B, T, d_model) tensors, one per layer
                       (after each transformer block, before final LN)
        logits: (B, T, n_tokens) final output logits
    """
    B, T = tokens.shape
    mask = torch.triu(torch.ones(T, T, device=tokens.device), diagonal=1).bool()

    x = model.token_embed(tokens)
    positions = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, -1)
    x = x + model.pos_embed(positions)

    layer_hiddens = []
    for layer in model.layers:
        x, _ = layer(x, mask)
        layer_hiddens.append(x.detach())

    final_h = model.ln_final(x)
    logits = model.output_proj(final_h)
    return layer_hiddens, logits


# ============================================================================
# Data generation helper (batched, with ground truth)
# ============================================================================

def generate_eval_data(p, pi, seq_len, n_eval, include_params=False):
    """Generate evaluation sequences with full Bayesian ground truth.

    Returns:
        all_tokens: list of n_eval token lists
        all_gt: list of n_eval ground truth lists
        all_params: list of (a, b) tuples if include_params, else None
    """
    from recurrence_bwt import recover_recurrence

    all_tokens = []
    all_gt = []
    all_params = [] if include_params else None

    for _ in range(n_eval):
        cfg = RecurrenceConfig(p=p, pi=pi, seq_len=seq_len)
        tokens, gt, metadata = generate_recurrence_sequence(cfg)
        all_tokens.append(tokens)
        all_gt.append(gt)

        if include_params:
            if metadata['true_class'] == 'program':
                ab = recover_recurrence(tokens[:3], p)
                all_params.append(ab if ab else (None, None))
            else:
                all_params.append((None, None))

    return all_tokens, all_gt, all_params


# ============================================================================
# Experiment 0: Cosine Similarity Across the Wall
# ============================================================================

def run_cosine_sim(args):
    """Compute per-layer cosine similarity between trained and untrained positions."""
    _sync_torch()
    ModelClass = _build_model_class()

    device = _resolve_device(args.device)
    p, pi, seq_len = args.p, args.pi, args.train_seq_len
    loss_horizon = args.loss_horizon

    # Load model
    model = ModelClass(
        vocab_size=p, n_tokens=p,
        d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_ff=args.d_ff,
        dropout=0.0, aux_classifier=False,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    n_layers = args.n_layers
    # Accumulate per-layer cosine similarities
    # cos_sim[layer][t] = list of cosine sims between h[ref_pos] and h[t]
    ref_pos = loss_horizon  # position 5 (last trained, 0-indexed)
    report_layers = [0, n_layers // 2, n_layers - 1]  # layers 0, 3, 5

    results = {layer_idx: {t: [] for t in range(1, seq_len)}
               for layer_idx in report_layers}

    # Also compute within-trained baseline: cos(h[ref_pos-1], h[ref_pos])
    within_trained = {layer_idx: [] for layer_idx in report_layers}

    batch_size = 64
    n_batches = (args.n_eval + batch_size - 1) // batch_size

    print(f"Computing cosine similarity at layers {report_layers}...")
    print(f"  Reference position: {ref_pos} (last trained)")
    print(f"  {args.n_eval} sequences in {n_batches} batches")

    for batch_idx in range(n_batches):
        n_this = min(batch_size, args.n_eval - batch_idx * batch_size)

        # Generate batch
        tokens_list = []
        for _ in range(n_this):
            cfg = RecurrenceConfig(p=p, pi=pi, seq_len=seq_len)
            tokens, _, _ = generate_recurrence_sequence(cfg)
            tokens_list.append(tokens)

        tokens_tensor = torch.tensor(tokens_list, dtype=torch.long).to(device)

        with torch.no_grad():
            layer_hiddens, _ = extract_per_layer_hiddens(model, tokens_tensor)

        for layer_idx in report_layers:
            h = layer_hiddens[layer_idx]  # (B, T, d_model)
            h_ref = h[:, ref_pos, :]  # (B, d_model)
            h_ref_norm = F.normalize(h_ref, dim=-1)

            # Within-trained baseline
            h_prev = h[:, ref_pos - 1, :]
            h_prev_norm = F.normalize(h_prev, dim=-1)
            cos_within = (h_ref_norm * h_prev_norm).sum(dim=-1)
            within_trained[layer_idx].extend(cos_within.cpu().tolist())

            for t in range(1, seq_len):
                h_t = h[:, t, :]
                h_t_norm = F.normalize(h_t, dim=-1)
                cos = (h_ref_norm * h_t_norm).sum(dim=-1)  # (B,)
                results[layer_idx][t].extend(cos.cpu().tolist())

    # Print results
    print(f"\n{'='*70}")
    print("COSINE SIMILARITY: h[ref_pos=5] vs h[t]")
    print(f"{'='*70}")

    output = {'ref_pos': ref_pos, 'layers': {}}

    for layer_idx in report_layers:
        wt_mean = np.mean(within_trained[layer_idx])
        print(f"\n  Layer {layer_idx} (within-trained cos(h[{ref_pos-1}], "
              f"h[{ref_pos}]): {wt_mean:.4f})")

        layer_data = {}
        for t in range(1, seq_len):
            mean_cos = np.mean(results[layer_idx][t])
            marker = " [T]" if t <= loss_horizon else " [U]"
            print(f"    t={t:2d}: cos={mean_cos:.4f}{marker}")
            layer_data[str(t)] = mean_cos

        layer_data['within_trained'] = wt_mean
        output['layers'][str(layer_idx)] = layer_data

    # Save
    out_path = os.path.join(args.output_dir, 'cosine_sim_results.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


# ============================================================================
# Experiment 1: Temperature Sweep
# ============================================================================

def run_temperature_sweep(args):
    """Post-hoc temperature rescaling at unrewarded positions."""
    _sync_torch()
    ModelClass = _build_model_class()

    device = _resolve_device(args.device)
    p, pi, seq_len = args.p, args.pi, args.train_seq_len
    loss_horizon = args.loss_horizon

    model = ModelClass(
        vocab_size=p, n_tokens=p,
        d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_ff=args.d_ff,
        dropout=0.0, aux_classifier=False,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Generate all data
    print(f"Generating {args.n_eval} evaluation sequences...")
    all_tokens, all_gt, _ = generate_eval_data(p, pi, seq_len, args.n_eval)

    # Batched model inference
    batch_size = 64
    all_logits = []
    for i in range(0, args.n_eval, batch_size):
        batch = all_tokens[i:i+batch_size]
        tokens_tensor = torch.tensor(batch, dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(tokens_tensor)
        all_logits.append(logits.cpu())
    all_logits = torch.cat(all_logits, dim=0).numpy()  # (n_eval, T, p)

    # Temperature grid
    temperatures = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]

    # === Top-1 accuracy at T=1.0 (no rescaling) ===
    print(f"\n{'='*70}")
    print("TOP-1 ACCURACY AT T=1.0 (NO RESCALING)")
    print(f"{'='*70}")

    top1_per_pos = {t: [] for t in range(1, seq_len)}
    rank_per_pos = {t: [] for t in range(1, seq_len)}
    top3_per_pos = {t: [] for t in range(1, seq_len)}
    margin_per_pos = {t: [] for t in range(1, seq_len)}  # margin on misses

    for i in range(args.n_eval):
        gt = all_gt[i]
        logits_i = all_logits[i]  # (T, p)

        for entry in gt:
            t = entry['t']
            if t < 1 or t >= seq_len:
                continue
            model_pos = t - 1

            logit_vec = logits_i[model_pos, :p]
            correct_token = all_tokens[i][t]

            # Top-1
            pred_token = int(np.argmax(logit_vec))
            top1_per_pos[t].append(1.0 if pred_token == correct_token else 0.0)

            # Rank of correct token (1-indexed, 1 = best)
            sorted_indices = np.argsort(-logit_vec)
            rank = int(np.where(sorted_indices == correct_token)[0][0]) + 1
            rank_per_pos[t].append(rank)

            # Top-3
            top3_per_pos[t].append(1.0 if rank <= 3 else 0.0)

            # Margin on misses
            if pred_token != correct_token:
                margin = logit_vec[pred_token] - logit_vec[correct_token]
                margin_per_pos[t].append(float(margin))

    for t in range(1, seq_len):
        if not top1_per_pos[t]:
            continue
        marker = " [T]" if t <= loss_horizon else " [U]"
        top1 = np.mean(top1_per_pos[t])
        mean_rank = np.mean(rank_per_pos[t])
        top3 = np.mean(top3_per_pos[t])
        miss_margin = np.mean(margin_per_pos[t]) if margin_per_pos[t] else 0.0
        print(f"  t={t:2d}: top1={top1:.3f}  rank={mean_rank:.2f}  "
              f"top3={top3:.3f}  miss_margin={miss_margin:.2f}{marker}")

    # === Per-temperature D_KL and MAE ===
    print(f"\n{'='*70}")
    print("TEMPERATURE SWEEP")
    print(f"{'='*70}")

    sweep_results = {}

    for T in temperatures:
        per_pos = {}
        for t in range(1, seq_len):
            per_pos[t] = {'dkl': [], 'mae': []}

        for i in range(args.n_eval):
            gt = all_gt[i]
            logits_i = all_logits[i]

            for entry in gt:
                t = entry['t']
                if t < 1 or t >= seq_len:
                    continue
                model_pos = t - 1

                logit_vec = logits_i[model_pos, :p]
                rescaled = logit_vec / T

                # Numerically stable softmax
                rescaled -= rescaled.max()
                exp_r = np.exp(rescaled)
                probs = exp_r / exp_r.sum()

                bayes_dist = entry['pred_dist']

                # Entropy
                H_model = -sum(probs[v] * np.log2(max(probs[v], 1e-10))
                               for v in range(p))
                H_bayes = entry['entropy']
                mae = abs(H_model - H_bayes)

                # D_KL(P_bayes || P_model)
                dkl = 0.0
                for v in range(p):
                    pb = bayes_dist.get(v, 0.0)
                    if pb > 0:
                        dkl += pb * math.log2(pb / max(probs[v], 1e-10))

                per_pos[t]['dkl'].append(dkl)
                per_pos[t]['mae'].append(mae)

        # Aggregate
        t_results = {}
        for t in range(1, seq_len):
            if per_pos[t]['dkl']:
                t_results[str(t)] = {
                    'dkl_mean': float(np.mean(per_pos[t]['dkl'])),
                    'mae_mean': float(np.mean(per_pos[t]['mae'])),
                }
        sweep_results[str(T)] = t_results

    # Find optimal T* per position
    print(f"\n{'='*70}")
    print("OPTIMAL T* PER POSITION")
    print(f"{'='*70}")

    optimal_T = {}
    for t in range(1, seq_len):
        best_T = None
        best_dkl = float('inf')
        for T in temperatures:
            t_str = str(t)
            T_str = str(T)
            if T_str in sweep_results and t_str in sweep_results[T_str]:
                dkl = sweep_results[T_str][t_str]['dkl_mean']
                if dkl < best_dkl:
                    best_dkl = dkl
                    best_T = T
        optimal_T[t] = best_T
        marker = " [T]" if t <= loss_horizon else " [U]"
        print(f"  t={t:2d}: T*={best_T}  D_KL={best_dkl:.4f}{marker}")

    # T* statistics for unrewarded positions
    unrewarded_Ts = [optimal_T[t] for t in range(loss_horizon + 1, seq_len)
                     if optimal_T[t] is not None]
    if unrewarded_Ts:
        print(f"\n  Unrewarded T* mean: {np.mean(unrewarded_Ts):.3f}")
        print(f"  Unrewarded T* std:  {np.std(unrewarded_Ts):.3f}")
        print(f"  Unrewarded T* range: [{min(unrewarded_Ts)}, {max(unrewarded_Ts)}]")

    # Save full results
    output = {
        'top1_per_pos': {str(t): float(np.mean(top1_per_pos[t]))
                         for t in range(1, seq_len) if top1_per_pos[t]},
        'rank_per_pos': {str(t): float(np.mean(rank_per_pos[t]))
                         for t in range(1, seq_len) if rank_per_pos[t]},
        'top3_per_pos': {str(t): float(np.mean(top3_per_pos[t]))
                         for t in range(1, seq_len) if top3_per_pos[t]},
        'margin_per_pos': {str(t): float(np.mean(margin_per_pos[t]))
                           for t in range(1, seq_len) if margin_per_pos[t]},
        'sweep_results': sweep_results,
        'optimal_T': {str(t): v for t, v in optimal_T.items()},
    }

    out_path = os.path.join(args.output_dir, 'temperature_sweep_results.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


# ============================================================================
# Experiment 2: Linear Probing
# ============================================================================

def run_probe(args):
    """Per-layer per-position linear probes."""
    _sync_torch()
    ModelClass = _build_model_class()

    device = _resolve_device(args.device)
    p, pi, seq_len = args.p, args.pi, args.train_seq_len
    loss_horizon = args.loss_horizon
    d_model = args.d_model
    n_layers = args.n_layers

    model = ModelClass(
        vocab_size=p, n_tokens=p,
        d_model=d_model, n_layers=n_layers,
        n_heads=args.n_heads, d_ff=args.d_ff,
        dropout=0.0, aux_classifier=False,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # === Collect hidden states and targets ===
    print(f"Collecting hidden states from {args.n_eval} sequences...")

    # Per-layer, per-position hidden state and target accumulation
    # layer_data[layer_idx][t] = {'hiddens': list of (d_model,), 'targets': list of int}
    layer_data = {layer_idx: {t: {'hiddens': [], 'tokens': [], 'params_a': [],
                                    'params_b': [], 'entropy': []}
                               for t in range(1, seq_len)}
                  for layer_idx in range(n_layers)}

    batch_size = 64
    n_collected = 0

    for batch_start in range(0, args.n_eval, batch_size):
        n_this = min(batch_size, args.n_eval - batch_start)

        tokens_list = []
        gt_list = []
        params_list = []

        for _ in range(n_this):
            cfg = RecurrenceConfig(p=p, pi=pi, seq_len=seq_len)
            tokens, gt, metadata = generate_recurrence_sequence(cfg)
            tokens_list.append(tokens)
            gt_list.append(gt)

            if metadata['true_class'] == 'program':
                from recurrence_bwt import recover_recurrence
                ab = recover_recurrence(tokens[:4], p)
                params_list.append(ab if ab else (None, None))
            else:
                params_list.append((None, None))

        tokens_tensor = torch.tensor(tokens_list, dtype=torch.long).to(device)

        with torch.no_grad():
            layer_hiddens, _ = extract_per_layer_hiddens(model, tokens_tensor)

        for layer_idx in range(n_layers):
            h = layer_hiddens[layer_idx].cpu().numpy()  # (B, T, d_model)

            for b in range(n_this):
                gt = gt_list[b]
                a_val, b_val = params_list[b]

                for entry in gt:
                    t = entry['t']
                    if t < 1 or t >= seq_len:
                        continue
                    model_pos = t - 1

                    layer_data[layer_idx][t]['hiddens'].append(h[b, model_pos])
                    layer_data[layer_idx][t]['tokens'].append(tokens_list[b][t])
                    layer_data[layer_idx][t]['params_a'].append(
                        a_val if a_val is not None else -1)
                    layer_data[layer_idx][t]['params_b'].append(
                        b_val if b_val is not None else -1)
                    layer_data[layer_idx][t]['entropy'].append(entry['entropy'])

        n_collected += n_this
        if n_collected % 512 == 0:
            print(f"  Collected {n_collected}/{args.n_eval}...")

    # === Train linear probes ===
    print(f"\nTraining linear probes ({args.probe_steps} steps each)...")

    probe_results = {}

    for layer_idx in range(n_layers):
        print(f"\n  Layer {layer_idx}:")
        layer_results = {}

        for t in range(1, seq_len):
            data = layer_data[layer_idx][t]
            if not data['hiddens']:
                continue

            X = np.array(data['hiddens'])  # (N, d_model)
            y_token = np.array(data['tokens'])  # (N,)
            y_entropy = np.array(data['entropy'])  # (N,)

            N = len(X)
            split = int(0.8 * N)

            X_train = torch.tensor(X[:split], dtype=torch.float32).to(device)
            X_test = torch.tensor(X[split:], dtype=torch.float32).to(device)

            # --- Probe 1: Next token classification ---
            y_train = torch.tensor(y_token[:split], dtype=torch.long).to(device)
            y_test = torch.tensor(y_token[split:], dtype=torch.long).to(device)

            probe = nn.Linear(d_model, p).to(device)
            opt = torch.optim.Adam(probe.parameters(), lr=1e-3)

            for step in range(args.probe_steps):
                # Mini-batch
                idx = torch.randint(0, len(X_train), (min(256, len(X_train)),))
                logits_p = probe(X_train[idx])
                loss = F.cross_entropy(logits_p, y_train[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()

            # Evaluate
            with torch.no_grad():
                test_logits = probe(X_test)
                test_pred = test_logits.argmax(dim=-1)
                token_acc = (test_pred == y_test).float().mean().item()

            # --- Probe 2: Bayesian entropy (regression) ---
            ye_train = torch.tensor(
                y_entropy[:split], dtype=torch.float32).to(device)
            ye_test = torch.tensor(
                y_entropy[split:], dtype=torch.float32).to(device)

            probe_e = nn.Linear(d_model, 1).to(device)
            opt_e = torch.optim.Adam(probe_e.parameters(), lr=1e-3)

            for step in range(args.probe_steps):
                idx = torch.randint(0, len(X_train), (min(256, len(X_train)),))
                pred_e = probe_e(X_train[idx]).squeeze(-1)
                loss_e = F.mse_loss(pred_e, ye_train[idx])
                opt_e.zero_grad()
                loss_e.backward()
                opt_e.step()

            with torch.no_grad():
                test_pred_e = probe_e(X_test).squeeze(-1)
                entropy_mse = F.mse_loss(test_pred_e, ye_test).item()
                entropy_r2 = 1.0 - (
                    ((test_pred_e - ye_test) ** 2).mean() /
                    ((ye_test - ye_test.mean()) ** 2).mean()
                ).item() if ye_test.std() > 0.01 else 0.0

            marker = " [T]" if t <= loss_horizon else " [U]"
            print(f"    t={t:2d}: token_acc={token_acc:.3f}  "
                  f"entropy_R2={entropy_r2:.3f}{marker}")

            layer_results[str(t)] = {
                'token_accuracy': token_acc,
                'entropy_mse': entropy_mse,
                'entropy_r2': entropy_r2,
                'n_train': split,
                'n_test': N - split,
            }

        probe_results[str(layer_idx)] = layer_results

    # Save
    output = {'checkpoint': args.checkpoint, 'probe_results': probe_results}
    out_path = os.path.join(args.output_dir, 'probe_results.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Probe experiment — calibration barrier diagnostics')

    parser.add_argument('--mode',
                        choices=['cosine_sim', 'temperature_sweep', 'probe'],
                        required=True)
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')

    # Task parameters
    parser.add_argument('--p', type=int, default=17)
    parser.add_argument('--pi', type=float, default=0.5)
    parser.add_argument('--train_seq_len', type=int, default=16)
    parser.add_argument('--loss_horizon', type=int, default=5)

    # Architecture (must match checkpoint)
    parser.add_argument('--d_model', type=int, default=192)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=6)
    parser.add_argument('--d_ff', type=int, default=768)

    # Evaluation
    parser.add_argument('--n_eval', type=int, default=2000)
    parser.add_argument('--probe_steps', type=int, default=2000,
                        help='Training steps for linear probes')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results/probe')

    args = parser.parse_args()
    _seed_all(args.seed)

    if args.mode == 'cosine_sim':
        run_cosine_sim(args)
    elif args.mode == 'temperature_sweep':
        run_temperature_sweep(args)
    elif args.mode == 'probe':
        run_probe(args)


if __name__ == '__main__':
    main()
