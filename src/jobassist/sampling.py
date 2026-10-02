"""Ex2: Next-token prediction & sampling mechanisms."""

from typing import Any

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute temperature-scaled softmax over 1D logits.

    Args:
        logits: 1D NumPy array of unnormalized token logits.
        temperature: Temperature parameter T >= 0.

    Returns:
        1D probability distribution summing to 1.0.
    """
    if temperature < 0:
        raise ValueError("Temperature must be non-negative (T >= 0).")

    if temperature == 0:
        # Greedy / argmax distribution
        probs = np.zeros_like(logits, dtype=np.float64)
        max_idx = np.argmax(logits)
        probs[max_idx] = 1.0
        return probs

    # Scale logits by temperature and apply max shift for numerical stability
    scaled_logits = logits.astype(np.float64) / temperature
    shifted = scaled_logits - np.max(scaled_logits)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits)
    return probs


def top_k_filter(logits: np.ndarray, k: int | None = None) -> np.ndarray:
    """Filter logits keeping only top k values, masking others with -inf.

    Args:
        logits: 1D NumPy array of unnormalized token logits.
        k: Number of top logits to keep. If None or k >= len(logits), returns logits unchanged.

    Returns:
        Filtered logits array.
    """
    if k is None or k <= 0 or k >= len(logits):
        return logits.copy()

    filtered = logits.copy()
    # Find cutoff threshold for top k elements
    top_k_indices = np.argpartition(logits, -k)[-k:]
    mask = np.ones_like(logits, dtype=bool)
    mask[top_k_indices] = False
    filtered[mask] = -np.inf
    return filtered


def top_p_filter(probs: np.ndarray, p: float = 1.0) -> np.ndarray:
    """Filter probability distribution using nucleus (top-p) sampling.

    Keeps the smallest prefix of descending-sorted tokens whose cumulative
    probability reaches or exceeds p.

    Args:
        probs: 1D probability distribution summing to 1.0.
        p: Target cumulative probability threshold (0.0 < p <= 1.0).

    Returns:
        Filtered probability distribution (un-renormalized, non-selected elements are 0.0).
    """
    if p >= 1.0:
        return probs.copy()
    if p <= 0:
        raise ValueError("Nucleus sampling threshold p must be > 0.")

    # Sort probabilities descending
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cum_probs = np.cumsum(sorted_probs)

    # Correct top-p searchsorted using side="left" and cutoff+1
    cutoff = int(np.searchsorted(cum_probs, p, side="left"))
    kept_indices = sorted_indices[: cutoff + 1]

    filtered_probs = np.zeros_like(probs)
    filtered_probs[kept_indices] = probs[kept_indices]
    return filtered_probs


def sample_next_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    rng: np.random.Generator | None = None,
) -> int:
    """Sample next token index following standard ordering:

    Temperature scaling -> Top-k filtering -> Softmax -> Top-p filtering -> Renormalization -> Sample.
    """
    if rng is None:
        rng = np.random.default_rng()

    if temperature < 0:
        raise ValueError("Temperature T must be >= 0.")

    if temperature == 0:
        return int(np.argmax(logits))

    # 1. Temperature scaling
    scaled_logits = logits.astype(np.float64) / temperature

    # 2. Top-k filter on logits
    if top_k is not None and top_k > 0:
        scaled_logits = top_k_filter(scaled_logits, top_k)

    # 3. Softmax to get probabilities
    # Shift by max of non-inf logits
    valid_mask = scaled_logits != -np.inf
    if not np.any(valid_mask):
        valid_mask = np.ones_like(scaled_logits, dtype=bool)

    max_val = np.max(scaled_logits[valid_mask])
    exp_logits = np.where(valid_mask, np.exp(scaled_logits - max_val), 0.0)
    denom = np.sum(exp_logits)
    probs = exp_logits / denom if denom > 0 else np.ones_like(logits) / len(logits)

    # 4. Top-p (nucleus) filter on probabilities
    if top_p < 1.0:
        probs = top_p_filter(probs, top_p)

    # 5. Renormalize probabilities
    total_p = np.sum(probs)
    if total_p > 0:
        probs = probs / total_p
    else:
        probs = np.ones_like(probs) / len(probs)

    # 6. Sample token index
    sampled_idx = rng.choice(len(probs), p=probs)
    return int(sampled_idx)


def compute_entropy(probs: np.ndarray) -> float:
    """Compute Shannon entropy H(P) in nats for probability distribution P."""
    non_zero = probs[probs > 0]
    return float(-np.sum(non_zero * np.log(non_zero)))


def run_sampling_demo(
    vocab: list[str],
    logits: np.ndarray,
    temperatures: list[float] | None = None,
    n_samples: int = 5000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Run empirical sampling demonstration across multiple temperature settings."""
    if temperatures is None:
        temperatures = [0.0, 0.3, 0.7, 1.0, 1.5]
    rng = np.random.default_rng(seed)
    results = []

    for T in temperatures:
        counts = {token: 0 for token in vocab}
        for _ in range(n_samples):
            idx = sample_next_token(logits, temperature=T, rng=rng)
            counts[vocab[idx]] += 1

        probs = np.array([counts[t] / n_samples for t in vocab])
        entropy = compute_entropy(probs)

        results.append({
            "temperature": T,
            "counts": counts,
            "probabilities": {t: round(p, 4) for t, p in zip(vocab, probs)},
            "entropy": round(entropy, 4),
        })

    return results
