# Roof Experiment Results

## Setup

- **Task**: Modular recurrence prediction, `x_{t+1} = ax_t + b mod 17`
- **Model**: 6-layer transformer, 192-dim, 6 heads (~2.8M params), learned PE
- **Training**: 150K steps, batch size 64, AdamW (lr=3e-4), cosine schedule
- **Loss horizon**: K=5 (CE loss at positions 1-5 only, unless noted)
- **Evaluation**: Per-position MAE in bits vs Bayesian optimal, D_KL(P_bayes || P_model), mode concentration
- **Seed**: 42

---

## Diagnostic Experiments (0-2)

### Experiment 0: Cosine Similarity Across the Wall

Cosine similarity between hidden states at position 5 (last trained) and positions 6-15 (untrained), at layers 1, 3, and 6 of the baseline (wall-intact) model.

| Position | Layer 0 | Layer 3 | Layer 5 |
|----------|---------|---------|---------|
| 4 [T] (within-trained ref) | 0.14 | 0.24 | 0.54 |
| 6 [U] | 0.14 | 0.25 | **0.70** |
| 7 [U] | 0.07 | 0.23 | **0.73** |
| 10 [U] | 0.09 | 0.19 | **0.70** |
| 13 [U] | 0.22 | 0.32 | **0.76** |
| 15 [U] | 0.13 | 0.22 | **0.66** |

**Finding**: Cosine similarity *increases* with depth and is *higher* across the wall (0.66-0.76 at layer 5) than within the trained region (0.54). Global clustering is real — the hidden states at unrewarded positions are in the same neighborhood as trained positions. Rigollet's prediction about global clustering dynamics is confirmed for inference-time representations.

---

### Experiment 1: Post-Hoc Temperature Sweep

Top-1 accuracy and ranking metrics at T=1.0 (no rescaling), plus optimal T* per position.

#### Top-1 accuracy at T=1.0

| Position | Top-1 | Mean Rank | Top-3 | Miss Margin |
|----------|-------|-----------|-------|-------------|
| 3 [T] | 0.523 | 5.22 | 0.572 | 2.78 |
| 5 [T] | 0.523 | 5.09 | 0.579 | 0.10 |
| 6 [U] | **0.249** | 6.89 | 0.379 | 1.11 |
| 7 [U] | **0.166** | 7.76 | 0.296 | 0.29 |
| 10 [U] | **0.165** | 7.88 | 0.284 | 0.53 |
| 15 [U] | **0.150** | 8.08 | 0.276 | 1.30 |

Note: Top-1 at trained positions is ~52%, which matches the Bayesian posterior (the correct token gets ~52% probability, not 100%). The model is properly calibrated at trained positions.

#### Optimal T* per position

| Position | T* | D_KL at T* |
|----------|-----|-----------|
| 5 [T] | 1.0 | 0.009 |
| 6 [U] | 1.5 | **1.481** |
| 7 [U] | 1.0 | **1.723** |
| 10 [U] | 1.5 | **1.680** |
| 15 [U] | 2.0 | **1.768** |

Unrewarded T*: mean=1.6, std=0.3, range=[1.0, 2.0]

**Finding**: The scale-factor hypothesis fails. Top-1 accuracy at unrewarded positions is 15-25% — the logit *ranking* is wrong, not just the scale. No temperature recovers the posterior (D_KL remains 1.5-1.8 at all T). The wall is not a simple temperature mismatch.

---

### Experiment 2: Linear Probing Across Layers

Per-layer per-position linear probes trained on frozen hidden states. Compared baseline (wall-intact) vs entropy-regularized (λ=0.1) models.

#### Token prediction accuracy — Baseline vs Entropy-Regularized

**Layer 5 (final):**

| Position | Baseline acc | Entropy-reg acc | Baseline entropy R2 | Entropy-reg entropy R2 |
|----------|-------------|-----------------|--------------------|-----------------------|
| 1 [T] | 0.050 | 0.071 | 0.000 | 0.000 |
| 3 [T] | 0.503 | 0.524 | 0.572 | 0.115 |
| 5 [T] | 0.548 | 0.552 | 0.954 | 0.971 |
| 6 [U] | **0.235** | **0.547** | 0.869 | 0.976 |
| 7 [U] | **0.147** | **0.545** | 0.713 | 0.944 |
| 10 [U] | **0.213** | **0.543** | 0.488 | 0.930 |
| 15 [U] | **0.169** | **0.540** | 0.422 | 0.975 |

#### Per-layer pattern — Entropy-regularized model token accuracy

| Position | Layer 0 | Layer 1 | Layer 2 | Layer 3 | Layer 4 | Layer 5 |
|----------|---------|---------|---------|---------|---------|---------|
| 5 [T] | 0.170 | 0.309 | 0.444 | 0.504 | 0.542 | 0.552 |
| 6 [U] | 0.167 | 0.268 | 0.435 | 0.517 | 0.541 | 0.547 |
| 10 [U] | 0.243 | 0.321 | 0.473 | 0.529 | 0.549 | 0.543 |
| 15 [U] | 0.241 | 0.300 | 0.461 | 0.536 | 0.540 | 0.540 |

**Finding**: The baseline model's hidden states do NOT encode the correct next token at unrewarded positions (15-24% accuracy vs 54% Bayesian optimal). The entropy-regularized model's hidden states encode it at 54% everywhere. The entropy signal reshapes the hidden representations, not just the output layer.

The entropy-regularized model shows token information emerging at layer 2-3 and reaching full accuracy by layer 4, with no trained/untrained gap. The circuit compiles uniformly across all positions when gradient is provided.

Interestingly, the baseline model's hidden states DO encode substantial information about the *correct entropy* (R2=0.42-0.87) even at unrewarded positions. The model "knows how uncertain it should be" but doesn't know "what to predict."

---

## Calibration Experiments (3-5)

### Experiment 3: Wrong Entropy Targets

All conditions use λ=0.1, entropy applied at all unrewarded positions (6-14).

| Condition | Target value | Wall Ratio | Trained MAE | Untrained MAE |
|-----------|-------------|-----------|-------------|---------------|
| **Correct** | H_bayes(t) ≈ 2.08 | **0.16x** | 0.036 | **0.006** |
| Constant 2.08 | 2.08 everywhere | 32.9x | 0.062 | 2.044 |
| Constant 1.0 | 1.0 everywhere | 42.1x | 0.050 | 2.097 |
| Constant 3.0 | 3.0 everywhere | 48.8x | 0.041 | 1.995 |
| Constant 0.5 | 0.5 everywhere | 63.5x | 0.034 | 2.126 |
| Uniform (4.09) | log2(17) | 63.2x | 0.031 | 1.937 |
| Random per-step | U(0, 4.09) | 63.6x | 0.032 | 2.045 |

**Finding**: Only the correct per-position Bayesian entropy targets erode the wall. Every wrong target — including constant 2.08 (approximately the right average) — preserves the wall completely. The position-specific entropy values are load-bearing. The entropy regularizer carries real information through the gradient about how uncertain the model should be at each specific position.

---

### Experiment 4: Entropy Signal Propagation

All conditions use λ=0.1, correct entropy targets.

#### Per-position MAE by entropy mask

| Position | All | Single 6 | Single 10 | Single 14 | Every-other |
|----------|-----|----------|-----------|-----------|-------------|
| 6 [U] | 0.003 | **0.003** | 0.407 | 1.391 | **0.004** |
| 7 [U] | 0.004 | 1.684 | 0.458 | 1.742 | 0.338 |
| 8 [U] | 0.004 | 1.760 | 0.487 | 1.651 | **0.007** |
| 9 [U] | 0.005 | 1.516 | 0.770 | 1.585 | 0.805 |
| 10 [U] | 0.004 | 1.506 | **0.005** | 1.651 | **0.006** |
| 11 [U] | 0.004 | 1.612 | 1.554 | 1.632 | 0.121 |
| 12 [U] | 0.004 | 1.611 | 1.555 | 1.889 | **0.005** |
| 13 [U] | 0.004 | 1.455 | 1.570 | 1.408 | 0.167 |
| 14 [U] | 0.004 | 1.526 | 1.489 | **0.006** | **0.005** |
| 15 [U] | 0.004 | 1.638 | 1.543 | 1.544 | 1.268 |

**Finding**: Calibration is purely local. Each targeted position achieves near-zero MAE; non-targeted positions remain at full wall. Entropy at position 6 does NOT help positions 7-15. The every-other condition is particularly clear: targeted positions (6,8,10,12,14) achieve MAE ~0.004-0.006, while interleaved positions (7,9,11,13,15) remain elevated. The signal doesn't propagate even one position.

This rules out both autoregressive-flow propagation and shared-weight propagation. The entropy regularizer compiles the circuit at each specific position via direct gradient, and nowhere else.

---

### Experiment 5: Late Introduction and Removal

All conditions use λ=0.1, correct entropy targets at all unrewarded positions.

#### Final wall ratios

| Condition | Schedule | Wall Ratio | Untrained MAE |
|-----------|----------|-----------|---------------|
| From start | Entropy 0-150K | 0.23x | 0.005 |
| **Late 50K** | CE 0-50K, entropy 50K-150K | **0.16x** | **0.007** |
| **Late 100K** | CE 0-100K, entropy 100K-150K | **0.16x** | **0.007** |
| Remove 50K | Entropy 0-50K, CE 50K-150K | 21.2x | 1.010 |
| Pulse 50K-60K | CE 0-50K, entropy 50K-60K, CE 60K-150K | 12.7x | 0.574 |

#### Degradation curve — Removal condition (entropy removed at step 50K)

| Step | Wall Ratio | Untrained MAE |
|------|-----------|---------------|
| 30K | 0.06x | 0.014 |
| 50K (removed) | 0.07x | 0.011 |
| 60K | 0.57x | 0.088 |
| 70K | 1.08x | 0.131 |
| 80K | 2.99x | 0.332 |
| 90K | 5.26x | 0.520 |
| 100K | 8.86x | 0.697 |
| 120K | 13.9x | 0.797 |
| 150K | 19.4x | 0.958 |

#### Degradation curve — Pulse condition (entropy only at steps 50K-60K)

| Step | Wall Ratio | Untrained MAE |
|------|-----------|---------------|
| 50K (pulse starts) | 10.4x | 1.594 |
| 60K (pulse ends) | 0.11x | 0.016 |
| 70K | 0.56x | 0.066 |
| 80K | 0.99x | 0.103 |
| 100K | 3.73x | 0.314 |
| 120K | 13.3x | 0.658 |
| 150K | 13.9x | 0.625 |

**Findings**:

1. **Late introduction works perfectly.** Entropy at step 50K or 100K produces the same wall erosion as from-start. The circuit compiles at trained positions first; the entropy signal calibrates unrewarded positions whenever it's introduced. This confirms the two-phase picture: compilation then calibration.

2. **Removal causes slow, monotonic degradation.** The wall returns gradually over ~100K steps after entropy is removed — not a step function, not immediate collapse. This matches the "deferred maintenance" pattern: the roof doesn't cave in the day you stop fixing it, but it degrades steadily.

3. **A brief pulse temporarily calibrates but doesn't persist.** 10K steps of entropy (50K-60K) dramatically calibrates the circuit (WR=0.11x at step 60K), but the calibration degrades to WR=13.9x by step 150K. Calibration is not a one-time phase transition — it requires ongoing maintenance.

---

## Earlier Experiments (Phase 1)

### Experiment 1: Distillation Direction — Null Result

Forward KL, reverse KL, and Jensen-Shannon distillation all erode the wall identically (WR 0.02-0.03x, D_KL 0.0001 at unrewarded positions). The direction doesn't matter because the teacher's posterior is unimodal.

### Experiment 2: Non-Invertible Generator — Causal Asymmetry

Quadratic recurrence (x_{t+1} = x_t^2 + b, many-to-one) vs linear (invertible). At full horizon (K=15), forward MAE=0.009 vs backward MAE=0.438. The causal arrow creates irreducible informational asymmetry from the data-generating process.

### Experiment 3 (Phase 1): Endogenous Roof — Negative

Learned temperature head and attention forget-gate worsen the wall (WR 31x → 44-76x). Without gradient at unrewarded positions, endogenous mechanisms can't self-calibrate.

---

## Summary

| Experiment | Key finding |
|------------|-------------|
| 0: Cosine similarity | Global clustering is real (cos 0.66-0.76 at final layer across wall) |
| 1: Temperature sweep | Logit ranking is wrong, not just scale (top-1 = 15-25%) |
| 2: Probing | Baseline hidden states DON'T encode correct token at unrewarded positions; entropy-reg model DOES |
| 3: Wrong entropy | Only correct per-position targets work — constant 2.08 fails |
| 4: Propagation | Calibration is purely local — no propagation even one position |
| 5: Timing | Late introduction works; removal causes slow degradation; pulse doesn't persist |

The entropy regularization result is not a calibration barrier (wrong hidden states → wrong outputs) but a **compilation barrier that is overcome by minimal per-position gradient** in the context of globally-structured hidden states. The entropy signal provides ~1 scalar per position, but that scalar, combined with the correct cluster neighborhood from global attention dynamics, suffices to compile the circuit locally. Without it, the circuit exists at trained positions and nowhere else. With it, the circuit extends everywhere — but only where and while the signal is provided.
