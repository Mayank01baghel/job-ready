"""Ex1: Token accounting & multilingual text analysis."""

import unicodedata
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import tiktoken


@dataclass
class TokenRow:
    language: str
    tokenizer: str
    text_sample: str
    char_count: int
    word_count: int
    token_count: int
    tokens_per_word: float
    chars_per_token: float
    token_ratio_vs_english: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_tokenizers() -> dict[str, Callable[[str], list[int]]]:
    """Return dictionary of available tokenizers."""
    tokenizers: dict[str, Callable[[str], list[int]]] = {}

    # Tiktoken cl100k_base
    try:
        enc_cl100k = tiktoken.get_encoding("cl100k_base")
        tokenizers["cl100k_base"] = lambda text: enc_cl100k.encode(text)
    except Exception:  # noqa: S110
        pass

    # Tiktoken gpt2 / r50k_base
    try:
        enc_gpt2 = tiktoken.get_encoding("gpt2")
        tokenizers["gpt2"] = lambda text: enc_gpt2.encode(text)
    except Exception:  # noqa: S110
        pass

    # Optional HuggingFace Tokenizer
    try:
        from transformers import AutoTokenizer  # type: ignore

        # Try a public non-gated model (e.g. gpt2 or distilbert)
        hf_tok = AutoTokenizer.from_pretrained("gpt2")
        tokenizers["hf_gpt2"] = lambda text: hf_tok.encode(text)
    except Exception:  # noqa: S110
        # HF transformers optional/unavailable or offline
        pass

    return tokenizers


def compare_token_counts(
    texts: dict[str, str],
    tokenizers: dict[str, Callable[[str], list[int]]] | None = None,
) -> list[TokenRow]:
    """Compare token counts across languages and tokenizers."""
    if tokenizers is None:
        tokenizers = get_tokenizers()

    rows: list[TokenRow] = []

    for tok_name, tok_fn in tokenizers.items():
        # Get baseline English token count for ratio calculation
        english_text = texts.get("english", "")
        english_tokens = len(tok_fn(english_text)) if english_text else 1

        for lang_name, text in texts.items():
            chars = len(text)
            words = len(text.split()) if text.split() else 1
            tokens = len(tok_fn(text))

            tokens_per_word = round(tokens / words, 3)
            chars_per_token = round(chars / max(tokens, 1), 3)
            ratio_vs_english = round(tokens / max(english_tokens, 1), 3)

            rows.append(
                TokenRow(
                    language=lang_name,
                    tokenizer=tok_name,
                    text_sample=text,
                    char_count=chars,
                    word_count=words,
                    token_count=tokens,
                    tokens_per_word=tokens_per_word,
                    chars_per_token=chars_per_token,
                    token_ratio_vs_english=ratio_vs_english,
                )
            )

    return rows


def check_unicode_normalization(
    text: str, tokenizer: Callable[[str], list[int]] | None = None
) -> dict[str, Any]:
    """Evaluate impact of NFC vs NFD normalization on tokenization counts."""
    if tokenizer is None:
        enc = tiktoken.get_encoding("cl100k_base")
        tokenizer = lambda t: enc.encode(t)

    nfc_text = unicodedata.normalize("NFC", text)
    nfd_text = unicodedata.normalize("NFD", text)

    nfc_tokens = len(tokenizer(nfc_text))
    nfd_tokens = len(tokenizer(nfd_text))

    return {
        "original_text": text,
        "nfc_char_count": len(nfc_text),
        "nfc_token_count": nfc_tokens,
        "nfd_char_count": len(nfd_text),
        "nfd_token_count": nfd_tokens,
        "counts_differ": nfc_tokens != nfd_tokens,
    }


def calculate_screening_cost(
    n: int = 1000,
    avg_tokens: int = 500,
    price_per_m: float = 0.15,
    hindi_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Calculate screening costs for N resumes with optional language overhead multiplier."""
    total_tokens = n * avg_tokens * hindi_multiplier
    total_cost = (total_tokens / 1_000_000.0) * price_per_m
    return {
        "num_resumes": n,
        "avg_tokens_per_resume": avg_tokens,
        "price_per_million_tokens": price_per_m,
        "hindi_multiplier": hindi_multiplier,
        "total_tokens": int(total_tokens),
        "total_cost_usd": round(total_cost, 6),
    }
