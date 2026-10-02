"""Tests for Exercise 1: Token accounting & multilingual text."""

from jobassist.tokens import (
    calculate_screening_cost,
    check_unicode_normalization,
    compare_token_counts,
    get_tokenizers,
)

SAMPLE_TEXTS = {
    "english": "Experienced software engineer specializing in scalable backend systems and distributed database architecture.",
    "devanagari_hindi": "अनुभवी सॉफ्टवेयर इंजीनियर जो बड़े पैमाने के बैकएंड सिस्टम और वितरित डेटाबेस आर्किटेक्चर में विशेषज्ञता रखते हैं।",
    "hinglish": "Anubhavi software engineer jo bade paimane ke backend systems aur vitrit database architecture mein visheshagyata rakhte hain.",
}


def test_tokenizers_available():
    tokenizers = get_tokenizers()
    assert "cl100k_base" in tokenizers
    assert "gpt2" in tokenizers


def test_compare_token_counts_ratios():
    rows = compare_token_counts(SAMPLE_TEXTS)
    assert len(rows) >= 6

    # Verify Devanagari tokens/word > English tokens/word for cl100k_base
    cl100k_rows = {r.language: r for r in rows if r.tokenizer == "cl100k_base"}
    assert "english" in cl100k_rows
    assert "devanagari_hindi" in cl100k_rows

    dev_row = cl100k_rows["devanagari_hindi"]
    eng_row = cl100k_rows["english"]

    assert dev_row.tokens_per_word > eng_row.tokens_per_word
    assert dev_row.token_ratio_vs_english > 1.0


def test_unicode_normalization():
    hindi_text = SAMPLE_TEXTS["devanagari_hindi"]
    norm_res = check_unicode_normalization(hindi_text)
    assert "nfc_token_count" in norm_res
    assert "nfd_token_count" in norm_res


def test_calculate_screening_cost():
    cost_std = calculate_screening_cost(n=1000, avg_tokens=500, price_per_m=0.15, hindi_multiplier=1.0)
    assert cost_std["total_tokens"] == 500000
    assert cost_std["total_cost_usd"] == 0.075

    cost_hindi = calculate_screening_cost(n=1000, avg_tokens=500, price_per_m=0.15, hindi_multiplier=2.5)
    assert cost_hindi["total_tokens"] == 1250000
    assert cost_hindi["total_cost_usd"] == 0.1875
