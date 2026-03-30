# Roof Experiment

Follow-up to the [wall-erosion experiment](https://github.com/riemannzeta/wall-erosion-experiment), probing the entropy regularization result that a scalar signal erodes the Shannon/Kolmogorov wall completely.

## Background

The wall-erosion experiment established that Misra's Shannon/Kolmogorov "wall" can be eroded by entropy regularization — a per-position scalar signal telling the model *how uncertain to be* at unrewarded positions. This is surprising: the signal specifies only a single number per position (the target entropy), yet the model responds by recovering the full 17-class Bayesian posterior. However, Experiment 3 below reveals this signal is not as information-poor as it first appears — the per-sequence Bayesian entropy target implicitly carries the program/random class distinction, which is the single highest-value bit in the task.

An initial hypothesis, grounded in [Rigollet's mean-field theory of transformers](https://arxiv.org/abs/2512.01868), predicted that this works because attention dynamics produce global clustering of token representations. The compiled circuit would exist everywhere in the hidden states, and the entropy signal would merely fix a scalar output-layer calibration parameter (the temperature). This "calibration barrier" hypothesis generated several testable predictions.

**The experiments falsified the simple calibration-barrier story but revealed something more interesting**: the entropy signal doesn't fix a scale factor — it provides the gradient signal needed for the circuit to compile at each unrewarded position individually. Calibration is local, requires the correct per-position entropy targets, and degrades gradually when the signal is removed.

## Results

Full results: [`results/RESULTS.md`](results/RESULTS.md)

### Experiments 0-2: Diagnosing the Mechanism

Three fast diagnostic experiments tested whether the wall is a calibration failure (correct hidden states, wrong output scale) or a compilation failure (hidden states don't encode the answer at unrewarded positions).

**Experiment 0 — Cosine Similarity**: Hidden states at unrewarded positions *are* in the same cluster as trained positions (cosine sim 0.65-0.76 at the final layer), and similarity *increases* with depth. Global clustering is real.

**Experiment 1 — Temperature Sweep**: Top-1 accuracy at unrewarded positions is only **15-25%** (chance = 6%). No post-hoc temperature recovers the Bayesian posterior — D_KL remains 1.5-1.8 bits at all temperatures. **The logit ranking is wrong**, not just the scale.

**Experiment 2 — Linear Probing**: The decisive result. Baseline model hidden states at unrewarded positions encode the correct token at only **15-24% accuracy** (probe). The entropy-regularized model's hidden states encode it at **54%** (matching Bayesian optimal). The entropy signal reshapes the hidden representations, not just the output layer.

| Position | Baseline probe acc | Entropy-reg probe acc | Baseline entropy R2 | Entropy-reg entropy R2 |
|----------|-------------------|----------------------|---------------------|----------------------|
| 5 [T] | 0.548 | 0.552 | 0.954 | 0.971 |
| 6 [U] | **0.235** | **0.547** | 0.869 | 0.976 |
| 10 [U] | **0.213** | **0.543** | 0.488 | 0.930 |
| 15 [U] | **0.169** | **0.540** | 0.422 | 0.975 |

### Experiment 3: Wrong Entropy Targets

**Only the correct per-position Bayesian entropy targets erode the wall.** Every wrong target preserves the wall, including constant 2.08 (approximately the right average value).

| Target | Wall Ratio | Untrained MAE |
|--------|-----------|---------------|
| Correct H_bayes(t) | **0.16x** | **0.006** |
| Constant 2.08 | 32.9x | 2.044 |
| Constant 1.0 | 42.1x | 2.097 |
| Constant 3.0 | 48.8x | 1.995 |
| Uniform (4.09) | 63.2x | 1.937 |
| Random | 63.6x | 2.045 |

The position-specific entropy values are load-bearing. The entropy regularizer is not a generic "scale anchor."

### Experiment 4: Signal Propagation

**Calibration is purely local.** Entropy at one position improves only that position. No forward, backward, or weight-mediated propagation.

| Entropy applied at | t=6 MAE | t=7 MAE | t=10 MAE | t=14 MAE |
|-------------------|---------|---------|----------|----------|
| All positions | 0.003 | 0.004 | 0.004 | 0.004 |
| Position 6 only | **0.003** | 1.684 | 1.506 | 1.526 |
| Position 10 only | 0.407 | 0.458 | **0.005** | 1.489 |
| Position 14 only | 1.391 | 1.742 | 1.651 | **0.006** |
| Every-other (6,8,10,12,14) | 0.004 | 0.338 | 0.006 | 0.005 |

### Experiment 5: Late Introduction and Removal

**Late introduction works perfectly** — entropy signal at step 50K or 100K produces the same wall erosion as from-start. The circuit compiles at trained positions first; calibration is added later.

**Removal causes slow degradation** — the maintenance interpretation is supported:

```
Step 50K (entropy removed): WR=0.07x   ← calibrated
Step 60K:                   WR=0.57x   ← beginning to degrade
Step 80K:                   WR=2.99x
Step 100K:                  WR=8.86x
Step 150K:                  WR=19.4x   ← wall mostly returned
```

**A 10K pulse temporarily calibrates but doesn't persist** (WR=0.11x at step 60K, degrades to WR=13.9x by step 150K). Calibration requires ongoing maintenance.

## Interpretation

The simple "calibration barrier" hypothesis — correct hidden states, wrong output temperature — is falsified. The hidden states at unrewarded positions do NOT encode the correct answer in the baseline model. But the experiments reveal a more nuanced picture:

1. **Global clustering is real** (Experiment 0): hidden states at unrewarded positions occupy the same region of representation space as trained positions, and increasingly so at deeper layers.

2. **The circuit doesn't compile without gradient** (Experiment 2): despite being in the right cluster, the hidden states at unrewarded positions don't carry the correct next-token prediction. Shared weight matrices and global attention aren't sufficient for the circuit to compile at positions that never receive gradient.

3. **The entropy signal provides per-position compilation gradient** (Experiments 3-4): only correct, position-specific entropy targets work. The signal doesn't propagate — each position needs its own gradient. The entropy regularizer is functioning as a per-position supervisory signal that compiles the circuit locally.

4. **Compilation and calibration are separable phases** (Experiment 5): the entropy signal can be introduced late (after the circuit exists at trained positions) and still works. But once established, the calibration degrades without ongoing maintenance — matching the [Maintaining Divergence](https://www.symmetrybroken.com/maintaining-divergence/) framework's prediction that synchronization costs must be continuously paid.

The deepest puzzle remains: *why does a per-position entropy target suffice to compile the circuit?* The signal provides ~1 scalar per position, yet the model recovers a full 17-class distribution. Experiment 3 sharpens this: the per-sequence Bayesian entropy implicitly carries the program/random class distinction (programs get near-zero entropy, randoms get near-maximum), so the signal is richer than "one number per position" — it's one *class-dependent* number per position. Still, that's far less than the full distribution.

The answer appears to involve the interaction between global structure and local gradient. Global clustering (Experiment 0) places the hidden states at unrewarded positions in the right *neighborhood* — cosine similarity 0.70 with trained positions. But the neighborhood is not the answer: the logit direction is wrong (Experiment 1, top-1 = 15-25%) and the hidden states don't encode the correct token (Experiment 2). The entropy gradient reshapes each hidden state from a partially-informative neighborhood into the correct configuration — a more substantial transformation than fixing a scale factor.

Experiment 2 also reveals an asymmetry between representation and optimization: the baseline model's hidden states *do* encode the correct entropy (R2 = 0.42-0.87) at unrewarded positions, even though they don't encode the correct token. The model partially "knows" how uncertain it should be, yet this representational knowledge doesn't compile the circuit. The entropy regularizer provides *gradient* based on that same information — and that's what works. Knowing is not the same as being optimized to act on what you know.

## Earlier Experiments (Phase 1)

Three earlier experiments tested KL direction, causal asymmetry, and endogenous mechanisms. These produced results explainable by simpler theories and motivated the sharper diagnostic experiments above.

- **Distillation direction** (null): Forward KL = Reverse KL = JS. Direction doesn't matter when the teacher's posterior is unimodal.
- **Non-invertible generator** (positive): Quadratic forward MAE=0.009 vs backward MAE=0.438 at full horizon. Causal asymmetry from the data-generating process is real and irreducible.
- **Endogenous roof** (negative): Learned temperature and forget-gate worsen the wall. Architectural affordances can't substitute for missing gradient.

## Reproducing

```bash
pip install -r requirements.txt

# Diagnostic experiments (fast, no training needed — requires baseline checkpoint)
python probe_experiment.py --mode cosine_sim --checkpoint <baseline_ckpt> --device cuda
python probe_experiment.py --mode temperature_sweep --checkpoint <baseline_ckpt> --device cuda
python probe_experiment.py --mode probe --checkpoint <baseline_ckpt> --device cuda

# Calibration experiments (Experiments 3-5)
python roof_experiment.py --experiment calibration --run_matrix \
    --seeds 42 --device cuda --output_dir results/roof_calibration
```

## Upstream

Extends the [wall-erosion experiment](https://github.com/riemannzeta/wall-erosion-experiment). The base task (modular linear recurrence wind tunnel) is from [vishalmisra/bayesian-wind-tunnel](https://github.com/vishalmisra/bayesian-wind-tunnel).

## License

MIT License. Upstream files subject to their respective license terms.
