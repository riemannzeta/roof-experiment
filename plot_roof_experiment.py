"""
Plotting for Roof Experiment Results
=====================================

Reads roof experiment summary JSON files and produces:

Experiment 1 (Distillation Direction):
  Figure 1: Per-position MAE for forward/reverse/JS distillation
  Figure 2: Per-position D_KL(P_bayes || P_model) — the key calibration plot
  Figure 3: Per-position entropy curves overlaid with Bayesian ground truth

Experiment 2 (Causal Direction):
  Figure 4: 4-panel linear/quadratic x forward/backward comparison
  Figure 5: Per-position entropy for quadratic forward vs backward

Usage:
    python plot_roof_experiment.py --experiment distill_direction \
        --results results/roof/distill_direction_summary.json

    python plot_roof_experiment.py --experiment causal_direction \
        --results results/roof/causal_direction_summary.json
"""

import json
import argparse
import os
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError:
    print("matplotlib required: pip install matplotlib")
    raise


# ============================================================================
# Color scheme (extends wall_erosion colors)
# ============================================================================

DIRECTION_COLORS = {
    'forward': '#2196F3',    # blue
    'reverse': '#E91E63',    # pink
    'js': '#FF9800',         # orange
}

DIRECTION_LABELS = {
    'forward': 'Forward KL',
    'reverse': 'Reverse KL',
    'js': 'Jensen-Shannon',
}

GENERATOR_COLORS = {
    'linear': '#4CAF50',     # green
    'quadratic': '#9C27B0',  # purple
}

PRED_DIR_STYLES = {
    'forward': '-',          # solid
    'backward': '--',        # dashed
}


def load_results(path):
    with open(path) as f:
        return json.load(f)


def group_results(results, group_keys):
    """Group results by specified keys, averaging over seeds."""
    groups = {}
    for r in results:
        key = tuple(r.get(k, None) for k in group_keys)
        if key not in groups:
            groups[key] = []
        groups[key].append(r)
    return groups


def _get_per_pos_arrays(result_list, metric='mae_mean'):
    """Extract per-position metric arrays, averaged across seeds."""
    all_positions = set()
    for r in result_list:
        all_positions.update(int(t) for t in r['per_position'].keys())

    positions = sorted(all_positions)
    values = []
    for t in positions:
        t_str = str(t)
        vals = [r['per_position'][t_str][metric]
                for r in result_list if t_str in r['per_position']]
        values.append(np.mean(vals) if vals else 0.0)

    return np.array(positions), np.array(values)


# ============================================================================
# Experiment 1 plots: Distillation Direction
# ============================================================================

def plot_distill_mae(results, output_dir, loss_horizon=5):
    """Figure 1: Per-position MAE for forward/reverse/JS distillation."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    groups = group_results(results,
                           ['distill_direction', 'control', 'subsidy_lambda'])

    # Plot baseline
    baseline = [r for r in results if r.get('condition', '').startswith('baseline')]
    if baseline:
        positions, maes = _get_per_pos_arrays(baseline)
        ax.plot(positions, maes, 'k-', linewidth=2, label='Baseline (wall)',
                alpha=0.8, zorder=10)

    # Plot each direction (active only, best lambda)
    for direction in ['forward', 'reverse', 'js']:
        # Find active conditions for this direction
        active = [r for r in results
                  if r.get('distill_direction') == direction
                  and not r.get('control', False)
                  and r.get('subsidy_lambda', 0) > 0]
        if not active:
            continue

        # Group by lambda, pick best (lowest untrained MAE)
        by_lambda = {}
        for r in active:
            lam = r['subsidy_lambda']
            if lam not in by_lambda:
                by_lambda[lam] = []
            by_lambda[lam].append(r)

        best_lam = min(by_lambda.keys(),
                       key=lambda l: np.mean([
                           r['wall_metrics']['untrained_mae']
                           for r in by_lambda[l]]))

        best = by_lambda[best_lam]
        positions, maes = _get_per_pos_arrays(best)
        color = DIRECTION_COLORS[direction]
        label = f"{DIRECTION_LABELS[direction]} (λ={best_lam})"
        ax.plot(positions, maes, color=color, linewidth=2,
                label=label, alpha=0.9)

        # Plot control (dashed)
        ctrl = [r for r in results
                if r.get('distill_direction') == direction
                and r.get('control', False)]
        if ctrl:
            positions_c, maes_c = _get_per_pos_arrays(ctrl)
            ax.plot(positions_c, maes_c, color=color, linewidth=1.5,
                    linestyle='--', alpha=0.5,
                    label=f"{DIRECTION_LABELS[direction]} (control)")

    ax.axvline(x=loss_horizon + 0.5, color='red', linestyle=':',
               alpha=0.5, label='Loss horizon')
    ax.set_xlabel('Position t', fontsize=12)
    ax.set_ylabel('MAE (bits)', fontsize=12)
    ax.set_title('Distillation Direction: Per-Position MAE', fontsize=14)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_ylim(bottom=-0.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'distill_direction_mae.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def plot_distill_dkl(results, output_dir, loss_horizon=5):
    """Figure 2: Per-position D_KL(P_bayes || P_model) — the key plot."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot baseline
    baseline = [r for r in results if r.get('condition', '').startswith('baseline')]
    if baseline:
        positions, dkls = _get_per_pos_arrays(baseline, metric='dkl_mean')
        ax.plot(positions, dkls, 'k-', linewidth=2, label='Baseline (wall)',
                alpha=0.8, zorder=10)

    for direction in ['forward', 'reverse', 'js']:
        active = [r for r in results
                  if r.get('distill_direction') == direction
                  and not r.get('control', False)
                  and r.get('subsidy_lambda', 0) > 0]
        if not active:
            continue

        by_lambda = {}
        for r in active:
            lam = r['subsidy_lambda']
            if lam not in by_lambda:
                by_lambda[lam] = []
            by_lambda[lam].append(r)

        best_lam = min(by_lambda.keys(),
                       key=lambda l: np.mean([
                           r['wall_metrics']['untrained_mae']
                           for r in by_lambda[l]]))

        best = by_lambda[best_lam]
        positions, dkls = _get_per_pos_arrays(best, metric='dkl_mean')
        color = DIRECTION_COLORS[direction]
        label = f"{DIRECTION_LABELS[direction]} (λ={best_lam})"
        ax.plot(positions, dkls, color=color, linewidth=2,
                label=label, alpha=0.9)

    ax.axvline(x=loss_horizon + 0.5, color='red', linestyle=':',
               alpha=0.5, label='Loss horizon')
    ax.set_xlabel('Position t', fontsize=12)
    ax.set_ylabel('D_KL(P_bayes || P_model) (bits)', fontsize=12)
    ax.set_title('Distillation Direction: Distribution Divergence from '
                 'Bayesian Posterior', fontsize=14)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_ylim(bottom=-0.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'distill_direction_dkl.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def plot_distill_entropy_curves(results, output_dir, loss_horizon=5):
    """Figure 3: Per-position entropy curves overlaid with Bayesian ground truth."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot Bayesian ground truth (from any result)
    any_result = results[0] if results else None
    if any_result:
        positions, h_bayes = _get_per_pos_arrays([any_result],
                                                  metric='H_bayes_mean')
        ax.plot(positions, h_bayes, 'k--', linewidth=2, alpha=0.6,
                label='Bayesian posterior', zorder=5)

    # Baseline model entropy
    baseline = [r for r in results if r.get('condition', '').startswith('baseline')]
    if baseline:
        positions, h_model = _get_per_pos_arrays(baseline,
                                                  metric='H_model_mean')
        ax.plot(positions, h_model, 'k-', linewidth=1.5, alpha=0.5,
                label='Baseline model')

    for direction in ['forward', 'reverse', 'js']:
        active = [r for r in results
                  if r.get('distill_direction') == direction
                  and not r.get('control', False)
                  and r.get('subsidy_lambda', 0) > 0]
        if not active:
            continue

        by_lambda = {}
        for r in active:
            lam = r['subsidy_lambda']
            if lam not in by_lambda:
                by_lambda[lam] = []
            by_lambda[lam].append(r)

        best_lam = min(by_lambda.keys(),
                       key=lambda l: np.mean([
                           r['wall_metrics']['untrained_mae']
                           for r in by_lambda[l]]))

        best = by_lambda[best_lam]
        positions, h_model = _get_per_pos_arrays(best,
                                                  metric='H_model_mean')
        color = DIRECTION_COLORS[direction]
        ax.plot(positions, h_model, color=color, linewidth=2,
                label=f"{DIRECTION_LABELS[direction]} (λ={best_lam})",
                alpha=0.9)

    ax.axvline(x=loss_horizon + 0.5, color='red', linestyle=':',
               alpha=0.5, label='Loss horizon')
    ax.set_xlabel('Position t', fontsize=12)
    ax.set_ylabel('Entropy (bits)', fontsize=12)
    ax.set_title('Distillation Direction: Entropy Tracking', fontsize=14)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'distill_direction_entropy.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def plot_distill_mode_concentration(results, output_dir, loss_horizon=5):
    """Figure 3b: Per-position mode concentration (max probability)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Bayesian mode concentration
    any_result = results[0] if results else None
    if any_result:
        positions, mc_bayes = _get_per_pos_arrays(
            [any_result], metric='mode_conc_bayes_mean')
        ax.plot(positions, mc_bayes, 'k--', linewidth=2, alpha=0.6,
                label='Bayesian posterior', zorder=5)

    for direction in ['forward', 'reverse', 'js']:
        active = [r for r in results
                  if r.get('distill_direction') == direction
                  and not r.get('control', False)
                  and r.get('subsidy_lambda', 0) > 0]
        if not active:
            continue

        by_lambda = {}
        for r in active:
            lam = r['subsidy_lambda']
            if lam not in by_lambda:
                by_lambda[lam] = []
            by_lambda[lam].append(r)

        best_lam = min(by_lambda.keys(),
                       key=lambda l: np.mean([
                           r['wall_metrics']['untrained_mae']
                           for r in by_lambda[l]]))

        best = by_lambda[best_lam]
        positions, mc = _get_per_pos_arrays(best,
                                            metric='mode_conc_model_mean')
        color = DIRECTION_COLORS[direction]
        ax.plot(positions, mc, color=color, linewidth=2,
                label=f"{DIRECTION_LABELS[direction]} (λ={best_lam})",
                alpha=0.9)

    ax.axvline(x=loss_horizon + 0.5, color='red', linestyle=':',
               alpha=0.5, label='Loss horizon')
    ax.set_xlabel('Position t', fontsize=12)
    ax.set_ylabel('Mode Concentration (max probability)', fontsize=12)
    ax.set_title('Distillation Direction: Mode Concentration', fontsize=14)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'distill_direction_mode_conc.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


# ============================================================================
# Experiment 2 plots: Causal Direction
# ============================================================================

def plot_causal_4panel(results, output_dir):
    """Figure 4: 4-panel linear/quadratic x forward/backward comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)

    panels = [
        (0, 0, 'linear', 'forward', 'Linear Forward'),
        (0, 1, 'linear', 'backward', 'Linear Backward'),
        (1, 0, 'quadratic', 'forward', 'Quadratic Forward'),
        (1, 1, 'quadratic', 'backward', 'Quadratic Backward'),
    ]

    for row, col, gen, direction, title in panels:
        ax = axes[row, col]

        # Find matching results (K=5 only for the 4-panel)
        matching = [r for r in results
                    if r.get('generator') == gen
                    and r.get('prediction_direction') == direction
                    and r.get('loss_horizon') == 5]

        if matching:
            # Plot Bayesian ground truth
            positions, h_bayes = _get_per_pos_arrays(matching,
                                                      metric='H_bayes_mean')
            ax.plot(positions, h_bayes, 'k--', linewidth=1.5, alpha=0.5,
                    label='Bayesian')

            # Plot model entropy
            positions, h_model = _get_per_pos_arrays(matching,
                                                      metric='H_model_mean')
            color = GENERATOR_COLORS[gen]
            ax.plot(positions, h_model, color=color, linewidth=2,
                    linestyle=PRED_DIR_STYLES[direction],
                    label='Model', alpha=0.9)

            # Wall ratio annotation
            wm = matching[0]['wall_metrics']
            ax.text(0.95, 0.95,
                    f"WR={wm['wall_ratio']:.1f}x",
                    transform=ax.transAxes, fontsize=11,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.axvline(x=5.5, color='red', linestyle=':', alpha=0.5)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes[1, 0].set_xlabel('Position t', fontsize=12)
    axes[1, 1].set_xlabel('Position t', fontsize=12)
    axes[0, 0].set_ylabel('Entropy (bits)', fontsize=12)
    axes[1, 0].set_ylabel('Entropy (bits)', fontsize=12)

    fig.suptitle('Causal Direction: Linear vs Quadratic, '
                 'Forward vs Backward', fontsize=15, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'causal_direction_4panel.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def plot_causal_mae_comparison(results, output_dir, loss_horizon=5):
    """Figure 5: Per-position MAE for all causal direction conditions."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    conditions = [
        ('linear', 'forward', 5),
        ('linear', 'backward', 5),
        ('quadratic', 'forward', 5),
        ('quadratic', 'backward', 5),
    ]

    for gen, direction, horizon in conditions:
        matching = [r for r in results
                    if r.get('generator') == gen
                    and r.get('prediction_direction') == direction
                    and r.get('loss_horizon') == horizon]

        if not matching:
            continue

        positions, maes = _get_per_pos_arrays(matching)
        color = GENERATOR_COLORS[gen]
        style = PRED_DIR_STYLES[direction]
        label = f"{gen.capitalize()} {direction}"
        ax.plot(positions, maes, color=color, linewidth=2,
                linestyle=style, label=label, alpha=0.9)

    ax.axvline(x=loss_horizon + 0.5, color='red', linestyle=':',
               alpha=0.5, label='Loss horizon')
    ax.set_xlabel('Position t', fontsize=12)
    ax.set_ylabel('MAE (bits)', fontsize=12)
    ax.set_title('Causal Direction: Per-Position MAE Comparison',
                 fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=-0.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'causal_direction_mae.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def plot_causal_full_horizon(results, output_dir):
    """Figure 6: Quadratic forward vs backward at FULL horizon (K=15).

    Tests whether backward fails even with gradient at all positions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for idx, (direction, title) in enumerate([
        ('forward', 'Quadratic Forward (K=15)'),
        ('backward', 'Quadratic Backward (K=15)'),
    ]):
        ax = axes[idx]
        matching = [r for r in results
                    if r.get('generator') == 'quadratic'
                    and r.get('prediction_direction') == direction
                    and r.get('loss_horizon') == 15]

        if matching:
            positions, h_bayes = _get_per_pos_arrays(matching,
                                                      metric='H_bayes_mean')
            ax.plot(positions, h_bayes, 'k--', linewidth=1.5, alpha=0.5,
                    label='Bayesian')

            positions, h_model = _get_per_pos_arrays(matching,
                                                      metric='H_model_mean')
            color = GENERATOR_COLORS['quadratic']
            ax.plot(positions, h_model, color=color, linewidth=2,
                    linestyle=PRED_DIR_STYLES[direction],
                    label='Model', alpha=0.9)

            wm = matching[0]['wall_metrics']
            ax.text(0.95, 0.95,
                    f"MAE={wm['trained_mae']:.3f}",
                    transform=ax.transAxes, fontsize=11,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Position t', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    axes[0].set_ylabel('Entropy (bits)', fontsize=12)
    fig.suptitle('Full Horizon: Does Backward Fail Even With Gradient '
                 'at All Positions?', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'causal_direction_full_horizon.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Plot roof experiment results')
    parser.add_argument('--experiment',
                        choices=['distill_direction', 'causal_direction'],
                        required=True)
    parser.add_argument('--results', type=str, required=True,
                        help='Path to summary JSON file')
    parser.add_argument('--output', type=str, default='figures',
                        help='Output directory for figures')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    results = load_results(args.results)

    if args.experiment == 'distill_direction':
        plot_distill_mae(results, args.output)
        plot_distill_dkl(results, args.output)
        plot_distill_entropy_curves(results, args.output)
        plot_distill_mode_concentration(results, args.output)
    elif args.experiment == 'causal_direction':
        plot_causal_4panel(results, args.output)
        plot_causal_mae_comparison(results, args.output)
        plot_causal_full_horizon(results, args.output)


if __name__ == '__main__':
    main()
