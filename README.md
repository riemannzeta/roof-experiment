# Wall Erosion Experiment

Testing the **synchronization tax prediction**: can the Shannon/Kolmogorov "wall" be eroded by providing indirect gradient flow to unrewarded positions?

## Background

[Misra (2025)](https://medium.com/@vishalmisra/the-wall-between-shannon-and-kolmogorov-65a9d7e8fb7c) gave a clean demonstration of the phenomenon: when a transformer is trained on modular linear recurrences (`x_{t+1} = ax_t + b mod 17`) with cross-entropy loss computed only at positions 1-K, the model achieves near-Bayesian performance at trained positions but fails catastrophically at untrained positions. Misra interprets this "wall" as an intrinsic boundary: LLMs compile localized prediction circuits based on pattern matching (Shannon) rather than learning generalized, position-independent algorithms (Kolmogorov).

This repository tests an alternative interpretation based on the [Maintaining Divergence](https://www.symmetrybroken.com/maintaining-divergence/#the-three-part-decomposition) framework, which predicted that the wall reflects where training allocates resources rather than an architectural ceiling.

**The Original Prediction:**

> If the wall simply reflects where synchronization costs are paid, providing a generic "maintenance subsidy" (indirect, non-task-specific gradient flow) to unrewarded positions should erode it.

**The Findings:**

The original prediction was too strong. Generic gradient flow — providing compute without task-relevant information — does not erode the wall. Matched controls confirm this cleanly: entropy regularization toward a uniform target, distillation from a random teacher, and hidden-state smoothness constraints all preserve the wall. Misra is right that generic compute is not enough.

But the wall is not as thick as the Shannon/Kolmogorov framing implies. Two mechanisms eliminate it completely:

- **Distillation** from a trained teacher erodes the wall by supplying the full Bayesian posterior at unrewarded positions. This supports Misra's interpretation: the teacher hands the student the answer.
- **Entropy regularization** erodes the wall by supplying only a scalar signal — *how uncertain to be* — without specifying *what to predict*. The model reconstructs the full posterior from its own internal representations given only this calibration hint. This is harder to reconcile with a hard Shannon/Kolmogorov divide, since the circuit capable of Bayesian inference already exists in the trained weights and needs only minimal supervisory signal to activate at new positions.

**Conclusion:**

The experiment disciplines the [Maintaining Divergence](https://www.symmetrybroken.com/maintaining-divergence/) framework: the "synchronization tax" cannot be paid with generic compute. Information content, not gradient flow, determines where circuits generalize. But the entropy regularization result suggests the wall may be a calibration barrier rather than a compilation barrier — the model possesses the circuit but cannot deploy it without a minimal task-relevant signal. Whether this distinction matters for practical LLM limitations remains an open question.

## Results

The wall **is not intrinsic**. Two mechanisms completely eliminate it:

| Condition | Trained MAE | Untrained MAE | Wall Ratio |
|-----------|------------|---------------|------------|
| Baseline-Horizon (the wall) | 0.247 | 1.755 | **7.1x** |
| A: Entropy regularization | 0.390 | **0.272** | **0.7x** |
| A: Entropy control (uniform) | 0.248 | 2.092 | 8.4x |
| B: Soft distillation | 0.225 | **0.045** | **0.2x** |
| B: Distill control (random) | 0.185 | 2.084 | 11.3x |

Matched controls that provide gradient flow but no task-relevant information preserve the wall, confirming the effect is driven by *information content*, not gradient flow alone.

Full results: [`results/RESULTS.md`](results/RESULTS.md)

### Per-position MAE

![Per-position MAE curves](figures/wall_erosion_per_position.png)

## Mechanisms tested

| Mechanism | What it provides at unrewarded positions | Wall erosion |
|-----------|----------------------------------------|-------------|
| **A: Entropy reg.** | Target entropy (how uncertain to be) | Complete |
| **B: Distillation** | Soft output distribution from trained teacher | Complete |
| **C: Smoothness** | Hidden-state continuity constraint | None (regularization artifact) |
| **D: Aux classifier** | Binary "is this a program?" signal | Modest |

Each mechanism has a matched control providing gradient flow with no task-relevant information.

## Reproducing

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Reproduce baselines (use --device mps on Apple Silicon, --device cuda on NVIDIA)
python wall_erosion_experiment.py --mechanism none --loss_horizon 15 \
    --n_steps 10000 --eval_every 5000 --device mps --seeds 42

python wall_erosion_experiment.py --mechanism none \
    --n_steps 10000 --eval_every 5000 --device mps --seeds 42

# Train teacher (for distillation)
python wall_erosion_experiment.py --train_teacher \
    --n_steps 10000 --eval_every 5000 --device mps --seeds 42

# Run a mechanism
python wall_erosion_experiment.py --mechanism entropy --subsidy_lambda 0.1 \
    --n_steps 10000 --eval_every 5000 --device mps --seeds 42

# Run with control
python wall_erosion_experiment.py --mechanism entropy --control --subsidy_lambda 0.1 \
    --n_steps 10000 --eval_every 5000 --device mps --seeds 42

# Full matrix (all mechanisms x controls x lambda sweep x 3 seeds)
python wall_erosion_experiment.py --run_matrix --seeds 42 43 44 --device mps
```

## Upstream

The base task (modular linear recurrence wind tunnel) is from [vishalmisra/bayesian-wind-tunnel](https://github.com/vishalmisra/bayesian-wind-tunnel). Files `recurrence_bwt.py` and `recurrence_extrapolation.py` are from that repo and provide data generation, Bayesian ground truth computation, and evaluation.

## License

This experiment code is released under the MIT License. The upstream files (`recurrence_bwt.py`, `recurrence_extrapolation.py`) are from [vishalmisra/bayesian-wind-tunnel](https://github.com/vishalmisra/bayesian-wind-tunnel) and are subject to its license terms.
