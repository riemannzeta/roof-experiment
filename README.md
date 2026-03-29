# Roof Experiment

Follow-up to the [wall-erosion experiment](https://github.com/riemannzeta/wall-erosion-experiment), testing whether KL divergence direction and causal asymmetry in the data-generating process determine how causal structure transfers in transformers.

## Background

The wall-erosion experiment established that Misra's Shannon/Kolmogorov "wall" is a **calibration barrier**, not a compilation barrier. The transformer globally compiles the modular recurrence rule, but its confidence collapses at out-of-distribution positions. Two mechanisms erode the wall completely: distillation from a trained teacher (forward KL) and entropy regularization (a scalar calibration signal).

This repository tests three follow-up questions:

1. **Does the direction of KL divergence matter?** Forward KL (mode-covering) vs reverse KL (mode-seeking) vs Jensen-Shannon (symmetric) distillation from the same teacher.
2. **Does genuine causal asymmetry in the data create a learnable asymmetry?** Non-invertible quadratic recurrence `x_{t+1} = x_t^2 + b mod 17` (many-to-one) vs invertible linear recurrence, forward vs backward prediction.
3. **Can the model learn to maintain its own calibration?** Learned temperature scaling and attention positional forget-gates, trained with no external subsidy.

## Results

Full results: [`results/RESULTS.md`](results/RESULTS.md)

### Experiment 1: Distillation Direction — Null Result

**The direction of KL divergence does not matter in this task.** Forward KL, reverse KL, and Jensen-Shannon distillation all erode the wall completely with identical calibration quality.

| Condition | Trained MAE | Untrained MAE | Wall Ratio | D_KL at t>5 |
|-----------|------------|---------------|------------|-------------|
| Baseline (wall) | 0.020 | 1.649 | **84x** | 1.5-1.7 |
| Forward KL (λ=0.5) | 0.033 | **0.001** | **0.02x** | 0.0001 |
| Reverse KL (λ=0.5) | 0.037 | **0.001** | **0.03x** | 0.0001 |
| Forward KL control (random teacher) | 0.027 | 2.009 | 75x | 2.2 |
| Reverse KL control (random teacher) | 0.027 | 2.009 | 75x | 2.2 |

The mode-covering vs mode-seeking distinction doesn't matter because the teacher's posterior at unrewarded positions is nearly unimodal. There is no multi-modal structure for reverse KL to collapse onto. What matters is having a *trained* teacher, not the direction of the divergence.

### Experiment 2: Non-Invertible Generator — Causal Asymmetry Confirmed

**The causal arrow in the data-generating process creates an irreducible informational asymmetry.**

| Condition | Trained MAE | Untrained MAE | Wall Ratio |
|-----------|------------|---------------|------------|
| Linear forward K=5 | 0.045 | 1.392 | 31x |
| Linear backward K=5 | 0.082 | 1.550 | 19x |
| Quadratic forward K=5 | 0.026 | 0.797 | 31x |
| Quadratic backward K=5 | 0.651 | 0.730 | 1.1x |
| **Quadratic forward K=15** | **0.009** | **—** | **—** |
| **Quadratic backward K=15** | **0.438** | **—** | **—** |

The cleanest result is the full-horizon (K=15) comparison, where both models receive gradient at every position:

- **Quadratic forward**: MAE=0.009, D_KL=0.0003 — near-perfect Bayesian tracking
- **Quadratic backward**: MAE=0.438, D_KL=2-3 bits — fundamentally cannot track the posterior

The backward model fails because the squaring map is 2-to-1: knowing `x_{t+1}` gives two possible values for `x_t`. This is an informational limit, not a computational one. No architecture can recover what the squaring destroyed.

The linear recurrence control confirms the setup: linear forward and backward both hit the wall similarly (WR 31x vs 19x), as expected for an invertible map with no genuine causal asymmetry.

### Experiment 3: Endogenous Roof — Negative Result

**The endogenous mechanisms failed to erode the wall.** They made it worse.

| Condition | Trained MAE | Untrained MAE | Wall Ratio |
|-----------|------------|---------------|------------|
| Baseline (no mechanism) | 0.047 | 1.462 | 31x |
| Learned temperature | 0.033 | 1.640 | 50x |
| Positional forget-gate | 0.037 | 1.636 | 44x |
| Temperature + gate | 0.023 | 1.728 | 76x |

The temperature head learned extreme values (T=20-40) — massively softening predictions everywhere rather than learning position-selective calibration. The gate values showed no trained/untrained differentiation (~0.3-0.6 uniformly). Without gradient signal at positions 6-15, these mechanisms have no information to calibrate against. The failure confirms that the wall is a *gradient signal* problem: endogenous architectural affordances cannot substitute for the missing supervisory signal.

## Reproducing

```bash
pip install -r requirements.txt

# Experiment 1: Distillation direction
python wall_erosion_experiment.py --train_teacher \
    --n_steps 150000 --device cuda --seeds 42
python roof_experiment.py --experiment distill_direction --run_matrix \
    --seeds 42 --device cuda --output_dir results/roof_distill

# Experiment 2: Causal direction
python roof_experiment.py --experiment causal_direction --run_matrix \
    --seeds 42 --device cuda --output_dir results/roof_causal

# Experiment 3: Endogenous roof
python roof_experiment.py --experiment endogenous_roof --run_matrix \
    --seeds 42 --device cuda --output_dir results/roof_endogenous

# Generate plots
python plot_roof_experiment.py --experiment distill_direction \
    --results results/roof_distill/distill_direction_summary.json
python plot_roof_experiment.py --experiment causal_direction \
    --results results/roof_causal/causal_direction_summary.json
python plot_roof_experiment.py --experiment endogenous_roof \
    --results results/roof_endogenous/endogenous_summary.json
```

## Upstream

Extends the [wall-erosion experiment](https://github.com/riemannzeta/wall-erosion-experiment). The base task (modular linear recurrence wind tunnel) is from [vishalmisra/bayesian-wind-tunnel](https://github.com/vishalmisra/bayesian-wind-tunnel). Files `recurrence_bwt.py` and `recurrence_extrapolation.py` are from that repo.

## License

MIT License. Upstream files subject to their respective license terms.
