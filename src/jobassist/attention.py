"""Ex3: Attention & context window mechanisms."""

import numpy as np


def create_prompt(system: str, jd: str, resume: str, question: str) -> dict[str, str]:
    """Structure system prompt, JD, resume, and question into XML formatted prompt payload."""
    full_prompt = (
        f"<system>{system.strip()}</system>\n"
        f"<job_description>{jd.strip()}</job_description>\n"
        f"<resume>{resume.strip()}</resume>\n"
        f"<question>{question.strip()}</question>"
    )
    return {
        "system": system,
        "job_description": jd,
        "resume": resume,
        "question": question,
        "full_prompt": full_prompt,
    }


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    causal: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute scaled dot-product attention with optional causal masking.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
        causal: If True, apply lower-triangular causal mask.

    Returns:
        Tuple of (Output matrix of shape (seq_len, d_v), Attention weights matrix of shape (seq_len, seq_len))
    """
    seq_len, d_k = Q.shape
    scores = np.dot(Q, K.T) / np.sqrt(d_k)

    if causal:
        # Create upper-triangular mask with -inf for future tokens
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        scores[mask] = -np.inf

    # Compute softmax over rows
    # Numerical stability
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    # Masked positions will have exp(-inf) = 0
    exp_scores[np.isnan(exp_scores)] = 0.0

    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
    attn_weights = np.where(sum_exp > 0, exp_scores / sum_exp, 0.0)

    output = np.dot(attn_weights, V)
    return output, attn_weights


def get_sinusoidal_positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Generate Sinusoidal Positional Encoding matrix of shape (seq_len, d_model)."""
    pe = np.zeros((seq_len, d_model), dtype=np.float64)
    position = np.arange(0, seq_len, dtype=np.float64)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, d_model, 2, dtype=np.float64) * -(np.log(10000.0) / d_model)
    )

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


def demonstrate_positional_sensitivity() -> dict[str, np.ndarray]:
    """Demonstrate that 'dog bites man' != 'man bites dog' only when positional encodings are added.

    Attention alone is permutation-equivariant (set-based operation).
    Adding positional encodings breaks permutation symmetry.
    """
    # 3 tokens: ["dog", "bites", "man"]
    d_model = 4
    rng = np.random.default_rng(42)

    # Base un-positioned embeddings for words
    emb_dog = np.array([1.0, 0.0, 0.5, 0.0])
    emb_bites = np.array([0.0, 2.0, 0.0, 1.0])
    emb_man = np.array([0.5, 0.0, 1.0, 0.0])

    # Sequence 1: "dog bites man"
    X1 = np.vstack([emb_dog, emb_bites, emb_man])
    # Sequence 2: "man bites dog" (permuting token positions 0 and 2)
    X2 = np.vstack([emb_man, emb_bites, emb_dog])

    # Linear projection matrices for Q, K, V
    W_q = rng.standard_normal((d_model, d_model))
    W_k = rng.standard_normal((d_model, d_model))
    W_v = rng.standard_normal((d_model, d_model))

    # 1. WITHOUT positional encoding (non-causal self-attention)
    Q1_nopo = np.dot(X1, W_q)
    K1_nopo = np.dot(X1, W_k)
    V1_nopo = np.dot(X1, W_v)
    out1_nopo, _ = scaled_dot_product_attention(Q1_nopo, K1_nopo, V1_nopo, causal=False)

    Q2_nopo = np.dot(X2, W_q)
    K2_nopo = np.dot(X2, W_k)
    V2_nopo = np.dot(X2, W_v)
    out2_nopo, _ = scaled_dot_product_attention(Q2_nopo, K2_nopo, V2_nopo, causal=False)

    # 2. WITH positional encoding
    PE = get_sinusoidal_positional_encoding(3, d_model)
    X1_po = X1 + PE
    X2_po = X2 + PE

    Q1_po = np.dot(X1_po, W_q)
    K1_po = np.dot(X1_po, W_k)
    V1_po = np.dot(X1_po, W_v)
    out1_po, _ = scaled_dot_product_attention(Q1_po, K1_po, V1_po, causal=False)

    Q2_po = np.dot(X2_po, W_q)
    K2_po = np.dot(X2_po, W_k)
    V2_po = np.dot(X2_po, W_v)
    out2_po, _ = scaled_dot_product_attention(Q2_po, K2_po, V2_po, causal=False)

    return {
        "out1_without_pos": out1_nopo,
        "out2_without_pos": out2_nopo,
        "out1_with_pos": out1_po,
        "out2_with_pos": out2_po,
        # Check equality: without position, token 1 ("bites") output is identical in both sequences
        "bites_equal_without_pos": np.allclose(out1_nopo[1], out2_nopo[1]),
        "bites_equal_with_pos": np.allclose(out1_po[1], out2_po[1]),
    }


def get_attention_explanation() -> str:
    """Return concise explanation of self-attention mechanics and complexity."""
    return (
        "Self-attention computes pairwise query-key similarity scores scaling as O(N^2) in time and memory "
        "relative to sequence length N. Without positional encodings, self-attention treats input sequences as permutation-invariant "
        "sets, producing identical representations for permuted tokens. Causal upper-triangular masking (-inf) guarantees "
        "that autoregressive tokens attend only to past and current positions."
    )
