"""
Quadratic Recurrence Module — Non-Invertible Generating Process
================================================================

Implements x_{t+1} = x_t^2 + b mod p, a many-to-one map that creates
genuine causal asymmetry: forward prediction is deterministic but backward
prediction is fundamentally ambiguous (quadratic residuosity).

This module parallels recurrence_bwt.py but for the quadratic case,
providing Bayesian ground truth for both forward and backward prediction.

The key mathematical difference from the linear case (x_{t+1} = ax_t + b):
  - Linear: invertible (a has multiplicative inverse mod p for a != 0)
  - Quadratic: non-invertible (x^2 is 2-to-1 since x^2 = (-x)^2 mod p)

For the Bayesian analysis:
  - Forward: b is identified from the first transition (k=2), making
    subsequent predictions deterministic. Bayes factor grows as p^(k-2).
  - Backward: even knowing b, predicting x_t from x_{t+1} requires
    solving x_t^2 = x_{t+1} - b mod p, which has 0 or 2 solutions.
"""

import math
import numpy as np


# ============================================================================
# Modular arithmetic helpers
# ============================================================================

def _mod_sqrt(c, p):
    """Find all x such that x^2 = c mod p (p prime).

    Returns a sorted list of solutions (0, 1, or 2 elements).
    Uses the Tonelli-Shanks algorithm for general primes,
    but for small p we just brute-force.
    """
    c = c % p
    if c == 0:
        return [0]
    # For small p, brute force is fine
    if p < 1000:
        roots = []
        for x in range(p):
            if (x * x) % p == c:
                roots.append(x)
        return sorted(roots)
    # Euler's criterion: c^((p-1)/2) = 1 mod p iff c is QR
    if pow(c, (p - 1) // 2, p) != 1:
        return []
    # Tonelli-Shanks
    if p % 4 == 3:
        r = pow(c, (p + 1) // 4, p)
        return sorted([r, p - r]) if r != p - r else [r]
    # General Tonelli-Shanks
    q, s = p - 1, 0
    while q % 2 == 0:
        q //= 2
        s += 1
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1
    m_val = s
    c_val = pow(z, q, p)
    t = pow(c, q, p)
    r = pow(c, (q + 1) // 2, p)
    while True:
        if t == 1:
            return sorted([r, p - r]) if r != p - r else [r]
        i = 1
        tmp = (t * t) % p
        while tmp != 1:
            tmp = (tmp * tmp) % p
            i += 1
        b_val = pow(c_val, 1 << (m_val - i - 1), p)
        m_val = i
        c_val = (b_val * b_val) % p
        t = (t * c_val) % p
        r = (r * b_val) % p


def _is_quadratic_residue(c, p):
    """Check if c is a quadratic residue mod p."""
    c = c % p
    if c == 0:
        return True
    return pow(c, (p - 1) // 2, p) == 1


# ============================================================================
# Sequence generation
# ============================================================================

def sample_quadratic_recurrence(p, seq_len=None):
    """Sample a quadratic recurrence: x_{t+1} = x_t^2 + b mod p.

    Args:
        p: prime modulus
        seq_len: number of tokens to generate (default: p + 1)

    Returns:
        b: the parameter
        seq: list of tokens [x_0, x_1, ..., x_{seq_len-1}]
    """
    if seq_len is None:
        seq_len = p + 1
    b = int(np.random.randint(0, p))
    x0 = int(np.random.randint(0, p))
    seq = [x0]
    x = x0
    for _ in range(seq_len - 1):
        x = (x * x + b) % p
        seq.append(x)
    return b, seq


# ============================================================================
# Bayesian inference — forward (causal direction)
# ============================================================================

def count_consistent_quadratic(seq, p):
    """Count b values consistent with x_{t+1} = x_t^2 + b mod p.

    Returns:
        k=0,1: p (any b is consistent with 0 or 1 observations)
        k>=2:  1 if consistent (b uniquely determined), 0 if falsified
    """
    k = len(seq)
    if k <= 1:
        return p
    # First transition determines b
    b_candidate = (seq[1] - seq[0] * seq[0]) % p
    # Check all subsequent transitions
    for t in range(1, k - 1):
        if (seq[t] * seq[t] + b_candidate) % p != seq[t + 1]:
            return 0
    return 1


def recover_quadratic_param(seq, p):
    """Recover b from a consistent quadratic sequence.

    Returns b if consistent (k >= 2), None otherwise.
    """
    if len(seq) < 2:
        return None
    b = (seq[1] - seq[0] * seq[0]) % p
    # Verify consistency
    for t in range(1, len(seq) - 1):
        if (seq[t] * seq[t] + b) % p != seq[t + 1]:
            return None
    return b


def bayesian_factor_quadratic(seq, p):
    """Bayes factor P(seq | H_P) / P(seq | H_R) for quadratic recurrence.

    Under H_P: x_0 ~ Uniform(Z_p), b ~ Uniform(Z_p),
               x_{t+1} = x_t^2 + b mod p.
    Under H_R: x_t ~ i.i.d. Uniform(Z_p).

    P(x_0,...,x_{k-1} | H_P) = (1/p) * (1/p) * prod_{t=0}^{k-2} 1
                               if consistent, since x_0 and b each uniform,
                               and subsequent values deterministic.
                             = 1/p^2 if consistent, 0 otherwise.
    P(x_0,...,x_{k-1} | H_R) = 1/p^k.

    So BF = p^{k-2} if consistent, 0 otherwise.
    For k=0,1: BF = 1 (indistinguishable).
    """
    k = len(seq)
    if k <= 1:
        return 1.0
    count = count_consistent_quadratic(seq, p)
    if count == 0:
        return 0.0
    # BF = p^(k-2) * count, but count is always 1 for k >= 2
    return float(p ** (k - 2)) * count


def class_posterior_quadratic(seq, p, pi):
    """P(H_P | x_0,...,x_{k-1}) for quadratic recurrence.

    Args:
        seq: observed prefix
        p: prime modulus
        pi: prior P(H_P)

    Returns: posterior probability of program hypothesis
    """
    bf = bayesian_factor_quadratic(seq, p)
    if bf == 0.0:
        return 0.0
    # w = pi * BF / (pi * BF + (1 - pi))
    numerator = pi * bf
    denominator = numerator + (1 - pi)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def bayesian_predictive_quadratic(seq, p, pi):
    """Bayesian predictive P(x_t | x_0,...,x_{t-1}) for quadratic recurrence.

    Under H_P: x_{t+1} = x_t^2 + b mod p, (x_0, b) ~ Uniform(Z_p^2).
    Under H_R: x_t ~ i.i.d. Uniform(Z_p).

    Returns: dict {value: probability} for all values in Z_p.
    """
    k = len(seq)
    w = class_posterior_quadratic(seq, p, pi)

    if k == 0:
        # Both predict uniform
        return {v: 1.0 / p for v in range(p)}

    if k == 1:
        # H_P: x_1 = x_0^2 + b. As b ranges over Z_p, x_1 is uniform.
        # H_R: uniform.
        return {v: 1.0 / p for v in range(p)}

    # k >= 2: b is determined by first transition
    count = count_consistent_quadratic(seq, p)
    if count == 0:
        # H_P falsified, only H_R contributes
        return {v: 1.0 / p for v in range(p)}

    b = (seq[1] - seq[0] * seq[0]) % p
    x_next = (seq[-1] * seq[-1] + b) % p

    # H_P predicts deterministically
    pred_hp = {v: (1.0 if v == x_next else 0.0) for v in range(p)}

    # Mixture
    pred = {}
    for v in range(p):
        pred[v] = w * pred_hp[v] + (1 - w) * (1.0 / p)
    return pred


# ============================================================================
# Bayesian inference — backward (anti-causal direction)
# ============================================================================

def count_consistent_quadratic_backward(rev_seq, p):
    """Count b values consistent with a REVERSED quadratic sequence.

    The reversed sequence [x_{T-1}, x_{T-2}, ..., x_0] was generated
    forward as x_{t+1} = x_t^2 + b. In reversed order, the "forward"
    relationship is: given rev_seq[i] = x_{T-1-i}, we have
    x_{T-1-i} = x_{T-2-i}^2 + b, i.e., rev_seq[i] = rev_seq[i+1]^2 + b.

    So b = rev_seq[i] - rev_seq[i+1]^2 mod p for any valid transition.
    """
    k = len(rev_seq)
    if k <= 1:
        return p
    # First "backward transition": rev_seq[0] = rev_seq[1]^2 + b
    # => b = rev_seq[0] - rev_seq[1]^2 mod p
    b_candidate = (rev_seq[0] - rev_seq[1] * rev_seq[1]) % p
    # Check all subsequent reversed transitions
    for i in range(1, k - 1):
        expected = (rev_seq[i + 1] * rev_seq[i + 1] + b_candidate) % p
        if expected != rev_seq[i]:
            return 0
    return 1


def bayesian_factor_quadratic_backward(rev_seq, p):
    """Bayes factor for backward (reversed) quadratic sequence.

    Under H_P (backward): the reversed sequence was generated by
    x_{t+1} = x_t^2 + b. For a backward predictor seeing
    [x_{T-1}, x_{T-2}, ...], to predict x_{T-k-1} (the "next" in
    reversed order), it needs to solve:
        x_{T-k-1}^2 + b = x_{T-k}
    i.e., x_{T-k-1}^2 = x_{T-k} - b mod p.

    The Bayes factor for detection is the same as forward (the
    statistical structure of transitions is symmetric for detection).
    But the PREDICTIVE distribution is fundamentally different.
    """
    k = len(rev_seq)
    if k <= 1:
        return 1.0
    count = count_consistent_quadratic_backward(rev_seq, p)
    if count == 0:
        return 0.0
    return float(p ** (k - 2)) * count


def class_posterior_quadratic_backward(rev_seq, p, pi):
    """P(H_P | reversed sequence) for backward prediction."""
    bf = bayesian_factor_quadratic_backward(rev_seq, p)
    if bf == 0.0:
        return 0.0
    numerator = pi * bf
    denominator = numerator + (1 - pi)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def bayesian_predictive_quadratic_backward(rev_seq, p, pi):
    """Bayesian predictive for BACKWARD prediction of quadratic recurrence.

    Given reversed sequence [x_{T-1}, x_{T-2}, ..., x_{T-k}],
    predict x_{T-k-1} (the "next" token in reversed order).

    The backward map requires solving x_{T-k-1}^2 = x_{T-k} - b mod p.
    This has 0 or 2 solutions (quadratic residuosity), creating
    genuine informational asymmetry.

    Returns: dict {value: probability} for all values in Z_p.
    """
    k = len(rev_seq)
    w = class_posterior_quadratic_backward(rev_seq, p, pi)

    if k == 0:
        return {v: 1.0 / p for v in range(p)}

    if k == 1:
        # H_P: can't determine b from one observation.
        # The "next" in reversed order is x_{T-2}, and x_{T-1} = x_{T-2}^2 + b.
        # As b ranges over Z_p, x_{T-2} could be any value whose square + b = x_{T-1}.
        # Marginalizing over b and x_{T-2}: uniform.
        return {v: 1.0 / p for v in range(p)}

    # k >= 2: b is determined
    count = count_consistent_quadratic_backward(rev_seq, p)
    if count == 0:
        return {v: 1.0 / p for v in range(p)}

    # Recover b: rev_seq[0] = rev_seq[1]^2 + b => b = rev_seq[0] - rev_seq[1]^2
    b = (rev_seq[0] - rev_seq[1] * rev_seq[1]) % p

    # To predict x_{T-k-1} (next in reversed order):
    # We need x_{T-k-1}^2 + b = rev_seq[k-1] (= x_{T-k})
    # i.e., x_{T-k-1}^2 = rev_seq[k-1] - b mod p
    target_sq = (rev_seq[k - 1] - b) % p
    roots = _mod_sqrt(target_sq, p)

    if len(roots) == 0:
        # No valid predecessor — this shouldn't happen for a genuine
        # quadratic sequence, but handle gracefully.
        # Under H_P, this is impossible, so H_P is falsified for prediction.
        pred_hp = {v: 1.0 / p for v in range(p)}
    elif len(roots) == 1:
        # Unique root (target_sq = 0)
        pred_hp = {v: (1.0 if v == roots[0] else 0.0) for v in range(p)}
    else:
        # TWO roots — the fundamental ambiguity!
        # Uniform over the two solutions
        pred_hp = {v: 0.0 for v in range(p)}
        for r in roots:
            pred_hp[r] = 1.0 / len(roots)

    # Mixture with random hypothesis
    pred = {}
    for v in range(p):
        pred[v] = w * pred_hp[v] + (1 - w) * (1.0 / p)
    return pred


# ============================================================================
# Entropy computation
# ============================================================================

def _predictive_entropy(pred_dist):
    """Shannon entropy in bits from a predictive distribution dict."""
    H = 0.0
    for v, prob in pred_dist.items():
        if prob > 0:
            H -= prob * math.log2(prob)
    return H


# ============================================================================
# Sequence generation (parallels recurrence_bwt.generate_recurrence_sequence)
# ============================================================================

class QuadraticConfig:
    """Configuration for quadratic recurrence experiments."""
    def __init__(self, p=17, pi=0.5, seq_len=16):
        self.p = p
        self.pi = pi
        self.seq_len = seq_len


def generate_quadratic_sequence(cfg, direction='forward'):
    """Generate a quadratic recurrence vs random sequence.

    With probability pi: draw b ~ Uniform(Z_p), x_0 ~ Uniform(Z_p),
                          then x_{t+1} = x_t^2 + b mod p.
    With probability 1-pi: draw each x_t ~ Uniform(Z_p) i.i.d.

    Args:
        cfg: QuadraticConfig
        direction: 'forward' or 'backward'. If 'backward', the sequence
                   is reversed before returning, and ground truth is
                   computed for backward prediction.

    Returns:
        tokens: list of token ids (length = seq_len)
        ground_truth: list of dicts with Bayesian ground truth at each position
        metadata: dict with sequence-level info
    """
    p = cfg.p
    seq_len = cfg.seq_len

    is_program = np.random.random() < cfg.pi

    if is_program:
        b, full_seq = sample_quadratic_recurrence(p, seq_len=seq_len)
        seq = full_seq[:seq_len]
        true_class = 'program'
    else:
        seq = [int(np.random.randint(0, p)) for _ in range(seq_len)]
        b = None
        true_class = 'random'

    if direction == 'backward':
        seq = list(reversed(seq))

    # Compute ground truth at each prediction position
    ground_truth = []
    for t in range(seq_len):
        prefix = seq[:t]
        if direction == 'forward':
            pred_dist = bayesian_predictive_quadratic(prefix, p, cfg.pi)
            w = class_posterior_quadratic(prefix, p, cfg.pi)
        else:
            pred_dist = bayesian_predictive_quadratic_backward(prefix, p, cfg.pi)
            w = class_posterior_quadratic_backward(prefix, p, cfg.pi)

        H = _predictive_entropy(pred_dist)

        ground_truth.append({
            't': t,
            'entropy': H,
            'pred_dist': pred_dist,
            'p_program': w,
        })

    metadata = {
        'true_class': true_class,
        'p': p,
        'b': b,
        'direction': direction,
        'vocab_size': p,
        'n_tokens': p,
        'header_len': 0,
    }

    return seq, ground_truth, metadata


# ============================================================================
# Linear recurrence backward (for the invertibility control)
# ============================================================================

def bayesian_predictive_linear_backward(rev_seq, p, pi):
    """Bayesian predictive for BACKWARD prediction of linear recurrence.

    The linear recurrence x_{t+1} = ax_t + b is invertible:
    x_t = a^{-1}(x_{t+1} - b) mod p (for a != 0).

    For reversed sequence [x_{T-1}, ..., x_{T-k}], predicting x_{T-k-1}:
    x_{T-k-1} = a^{-1}(x_{T-k} - b) mod p.

    This is deterministic (1-to-1) once (a, b) are identified,
    in contrast to the quadratic case which is 2-to-1.
    """
    # The reversed linear sequence has the SAME statistical structure
    # as a forward linear sequence (just with different parameters).
    # If x_{t+1} = ax_t + b, then x_t = a'x_{t+1} + b' where
    # a' = a^{-1}, b' = -a^{-1}b.
    # So the reversed sequence IS a linear recurrence with parameters (a', b').
    # Import from recurrence_bwt to reuse existing machinery.
    from recurrence_bwt import bayesian_predictive_recurrence
    return bayesian_predictive_recurrence(rev_seq, p, pi)


# ============================================================================
# Unit tests
# ============================================================================

def _run_tests():
    """Run unit tests for quadratic recurrence module."""
    print("Running quadratic recurrence unit tests...")
    p = 17
    pi = 0.5
    errors = 0

    # Test 1: Forward prediction is deterministic for k >= 2
    print("\n  Test 1: Forward prediction deterministic for k >= 2")
    for trial in range(20):
        b, seq = sample_quadratic_recurrence(p, seq_len=8)
        for k in range(2, 7):
            pred = bayesian_predictive_quadratic(seq[:k], p, pi)
            actual_next = seq[k]
            # After k=2, b is identified. The posterior weight w should be
            # high and the predicted distribution should be peaked.
            w = class_posterior_quadratic(seq[:k], p, pi)
            if k >= 3 and w < 0.9:
                print(f"    FAIL: k={k}, w={w:.4f} (expected > 0.9)")
                errors += 1
            if k >= 3:
                # Check that the highest-probability prediction is correct
                best_v = max(pred, key=pred.get)
                if best_v != actual_next:
                    print(f"    FAIL: k={k}, predicted {best_v}, actual {actual_next}")
                    errors += 1
    if errors == 0:
        print("    PASS")

    # Test 2: Backward prediction has genuine 2-to-1 ambiguity
    print("\n  Test 2: Backward prediction is ambiguous (2-to-1)")
    ambiguous_count = 0
    for trial in range(50):
        b, seq = sample_quadratic_recurrence(p, seq_len=8)
        rev_seq = list(reversed(seq))
        for k in range(3, 7):
            pred = bayesian_predictive_quadratic_backward(rev_seq[:k], p, pi)
            # Count how many values have significant probability under H_P
            w = class_posterior_quadratic_backward(rev_seq[:k], p, pi)
            if w > 0.5:
                high_prob = [v for v, prob in pred.items() if prob > 0.1]
                if len(high_prob) == 2:
                    ambiguous_count += 1
    print(f"    Ambiguous predictions (2 modes): {ambiguous_count} / 200")
    if ambiguous_count > 30:
        print("    PASS (significant fraction show 2-to-1 ambiguity)")
    else:
        print("    WARN: fewer ambiguous cases than expected")

    # Test 3: Bayes factor grows correctly
    print("\n  Test 3: Bayes factor grows as p^(k-2)")
    b, seq = sample_quadratic_recurrence(p, seq_len=8)
    for k in range(1, 8):
        bf = bayesian_factor_quadratic(seq[:k], p)
        expected = p ** max(0, k - 2) if count_consistent_quadratic(seq[:k], p) > 0 else 0
        if abs(bf - expected) > 0.01:
            print(f"    FAIL: k={k}, BF={bf}, expected={expected}")
            errors += 1
        else:
            print(f"    k={k}: BF={bf:.0f} (expected {expected})")
    if errors == 0:
        print("    PASS")

    # Test 4: Consistency of count functions
    print("\n  Test 4: count_consistent_quadratic correctness")
    for trial in range(20):
        b, seq = sample_quadratic_recurrence(p, seq_len=8)
        for k in range(1, 8):
            c = count_consistent_quadratic(seq[:k], p)
            if k <= 1 and c != p:
                print(f"    FAIL: k={k}, count={c}, expected {p}")
                errors += 1
            elif k >= 2 and c != 1:
                print(f"    FAIL: k={k}, count={c}, expected 1 (seq={seq[:k]})")
                errors += 1
    # Test with corrupted sequence
    b, seq = sample_quadratic_recurrence(p, seq_len=8)
    corrupted = list(seq)
    corrupted[3] = (corrupted[3] + 1) % p  # break it
    c = count_consistent_quadratic(corrupted[:5], p)
    if c != 0:
        print(f"    FAIL: corrupted sequence count={c}, expected 0")
        errors += 1
    else:
        print("    PASS (including corrupted sequence test)")

    # Test 5: mod_sqrt correctness
    print("\n  Test 5: _mod_sqrt correctness for p=17")
    for c in range(p):
        roots = _mod_sqrt(c, p)
        for r in roots:
            if (r * r) % p != c:
                print(f"    FAIL: sqrt({c}) mod {p} gave {r}, "
                      f"but {r}^2 = {(r*r)%p}")
                errors += 1
        # Check completeness
        actual_roots = sorted([x for x in range(p) if (x * x) % p == c])
        if roots != actual_roots:
            print(f"    FAIL: sqrt({c}) mod {p}: got {roots}, "
                  f"expected {actual_roots}")
            errors += 1
    qr_count = sum(1 for c in range(1, p) if _is_quadratic_residue(c, p))
    print(f"    Quadratic residues mod {p}: {qr_count} (expected {(p-1)//2}={p//2})")
    if qr_count == (p - 1) // 2:
        print("    PASS")
    else:
        errors += 1

    # Test 6: Entropy comparison — forward vs backward
    print("\n  Test 6: Forward entropy < backward entropy (for identified programs)")
    forward_entropies = []
    backward_entropies = []
    for trial in range(100):
        b, seq = sample_quadratic_recurrence(p, seq_len=8)
        rev_seq = list(reversed(seq))
        # At k=4, program should be well-identified
        H_fwd = _predictive_entropy(
            bayesian_predictive_quadratic(seq[:4], p, pi))
        H_bwd = _predictive_entropy(
            bayesian_predictive_quadratic_backward(rev_seq[:4], p, pi))
        forward_entropies.append(H_fwd)
        backward_entropies.append(H_bwd)
    mean_fwd = np.mean(forward_entropies)
    mean_bwd = np.mean(backward_entropies)
    print(f"    Mean forward entropy  (k=4): {mean_fwd:.4f} bits")
    print(f"    Mean backward entropy (k=4): {mean_bwd:.4f} bits")
    if mean_fwd < mean_bwd:
        print("    PASS (forward < backward, as expected for non-invertible map)")
    else:
        print("    FAIL: expected forward entropy < backward entropy")
        errors += 1

    # Test 7: generate_quadratic_sequence smoke test
    print("\n  Test 7: generate_quadratic_sequence smoke test")
    cfg = QuadraticConfig(p=17, pi=0.5, seq_len=16)
    for direction in ['forward', 'backward']:
        tokens, gt, meta = generate_quadratic_sequence(cfg, direction=direction)
        assert len(tokens) == 16, f"Expected 16 tokens, got {len(tokens)}"
        assert len(gt) == 16, f"Expected 16 gt entries, got {len(gt)}"
        assert meta['direction'] == direction
        assert all(0 <= t < p for t in tokens), "Token out of range"
        for entry in gt:
            assert abs(sum(entry['pred_dist'].values()) - 1.0) < 1e-6, \
                f"pred_dist doesn't sum to 1 at t={entry['t']}"
    print("    PASS")

    print(f"\n{'='*50}")
    if errors == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"FAILURES: {errors}")
    print(f"{'='*50}")
    return errors


if __name__ == '__main__':
    _run_tests()
