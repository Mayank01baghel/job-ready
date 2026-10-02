"""Run all 9 exercises, print tables, and write results to results/*.json."""

import json
import os

import numpy as np

from jobassist.attention import demonstrate_positional_sensitivity, get_attention_explanation
from jobassist.context import simulate_lost_in_the_middle_recall
from jobassist.cost import TokenPricing, generate_cost_comparison_table
from jobassist.hallucination import (
    BAD_HALLUCINATED_ASSESSMENT,
    GOOD_ASSESSMENT,
    SARAH_CHEN_RESUME,
    verify_grounding,
)
from jobassist.llm import MockLLM
from jobassist.sampling import run_sampling_demo
from jobassist.schema import validate_json_output
from jobassist.tiers import (
    simulate_dynamic_routing,
)
from jobassist.tokens import check_unicode_normalization, compare_token_counts
from jobassist.training_stages import TrainingStage, explain

SAMPLE_TEXTS = {
    "english": "Experienced software engineer specializing in scalable backend systems and distributed database architecture.",
    "devanagari_hindi": "अनुभवी सॉफ्टवेयर इंजीनियर जो बड़े पैमाने के बैकएंड सिस्टम और वितरित डेटाबेस आर्किटेक्चर में विशेषज्ञता रखते हैं।",
    "hinglish": "Anubhavi software engineer jo bade paimane ke backend systems aur vitrit database architecture mein visheshagyata rakhte hain.",
}


def ensure_results_dir():
    os.makedirs("results", exist_ok=True)


def write_result(filename: str, data: dict):
    path = os.path.join("results", filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"-> Wrote {path}\n")


def print_table(title, rows, keys):
    print(f"\n=== {title} ===")
    if not rows:
        return
        
    # Find column widths
    widths = {k: len(str(k)) for k in keys}
    for row in rows:
        for k in keys:
            widths[k] = max(widths[k], len(str(row.get(k, ""))))
            
    header = " | ".join(f"{k!s:<{widths[k]}}" for k in keys)
    print(header)
    print("-" * len(header))
    
    for row in rows:
        print(" | ".join(f"{row.get(k, '')!s:<{widths[k]}}" for k in keys))
    print()


def run_ex1_tokens():
    print("--- Running Ex1: Token Accounting ---")
    rows = compare_token_counts(SAMPLE_TEXTS)
    dict_rows = [r.to_dict() for r in rows]
    
    keys = ["tokenizer", "language", "token_count", "tokens_per_word", "chars_per_token", "token_ratio_vs_english"]
    print_table("Token Comparison", dict_rows, keys)
    
    norm = check_unicode_normalization(SAMPLE_TEXTS["devanagari_hindi"])
    print(f"Unicode Normalization Differs? {norm['counts_differ']} (NFC: {norm['nfc_token_count']}, NFD: {norm['nfd_token_count']})")
    
    write_result("ex1_tokens.json", {"comparisons": dict_rows, "normalization": norm})


def run_ex2_sampling():
    print("--- Running Ex2: Sampling ---")
    vocab = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    import numpy as np
    logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.1, -0.5, -1.0, -2.0])
    
    results = run_sampling_demo(vocab, logits)
    
    table = []
    for r in results:
        table.append({
            "temperature": r["temperature"],
            "entropy": r["entropy"],
            "prob_a": r["probabilities"]["a"],
            "prob_j": r["probabilities"]["j"]
        })
    print_table("Sampling Distributions", table, ["temperature", "entropy", "prob_a", "prob_j"])
    
    write_result("ex2_sampling.json", {"demo_results": results})


def run_ex3_attention():
    print("--- Running Ex3: Attention ---")
    demo = demonstrate_positional_sensitivity()
    
    table = [
        {"configuration": "Without Positional Encoding", "token_1_equal": bool(demo["bites_equal_without_pos"])},
        {"configuration": "With Positional Encoding", "token_1_equal": bool(demo["bites_equal_with_pos"])},
    ]
    print_table("Positional Sensitivity", table, ["configuration", "token_1_equal"])
    print(get_attention_explanation())
    
    # Can't json serialize numpy arrays easily, so we just write boolean flags
    res = {k: bool(v) for k, v in demo.items() if isinstance(v, (bool, np.bool_))}
    write_result("ex3_attention.json", res)


def run_ex4_context():
    print("--- Running Ex4: Lost-in-the-Middle ---")
    llm = MockLLM()
    results = simulate_lost_in_the_middle_recall(llm)
    
    stuffed = results["stuffed_context_recall"]
    retrieved = results["retrieved_context_recall"]
    
    table = []
    for pos in stuffed:
        table.append({
            "position": pos,
            "stuffed_recall": stuffed[pos],
            "retrieved_recall": retrieved[pos]
        })
    print_table("Recall by Strategy and Position", table, ["position", "stuffed_recall", "retrieved_recall"])
    
    write_result("ex4_context.json", results)


def run_ex5_training():
    print("--- Running Ex5: Training Stages ---")
    stages = [TrainingStage.PRETRAINING, TrainingStage.SUPERVISED_FINE_TUNING, TrainingStage.PREFERENCE_TUNING]
    
    table = []
    for st in stages:
        exp = explain(st)
        table.append({
            "stage": st.value,
            "objective": exp["objective"]
        })
    print_table("Training Stages", table, ["stage", "objective"])
    
    res = {st.value: explain(st) for st in stages}
    write_result("ex5_training.json", res)


def run_ex6_tiers():
    print("--- Running Ex6: Model Tiers ---")
    results = simulate_dynamic_routing()
    
    table = [
        {"metric": "Total Queries", "value": results["total_queries"]},
        {"metric": "All Large Cost", "value": f"${results['all_large_baseline_cost']:.2f}"},
        {"metric": "Blended Routing Cost", "value": f"${results['blended_cost']:.2f}"},
        {"metric": "Savings", "value": f"${results['savings_usd']:.2f} ({results['savings_percent']:.1f}%)"}
    ]
    print_table("Router Simulation", table, ["metric", "value"])
    
    write_result("ex6_tiers.json", results)


def run_ex7_hallucination():
    print("--- Running Ex7: Hallucination ---")
    good_report = verify_grounding(GOOD_ASSESSMENT, SARAH_CHEN_RESUME)
    bad_report = verify_grounding(BAD_HALLUCINATED_ASSESSMENT, SARAH_CHEN_RESUME)
    
    table = [
        {"assessment": "Good", "passed": good_report.passed, "errors": len(good_report.unsupported_claims) + len(good_report.missing_evidence)},
        {"assessment": "Bad (Hallucinated)", "passed": bad_report.passed, "errors": len(bad_report.unsupported_claims) + len(bad_report.missing_evidence)},
    ]
    print_table("Grounding Verification", table, ["assessment", "passed", "errors"])
    
    write_result("ex7_hallucination.json", {
        "good_assessment_errors": good_report.unsupported_claims + good_report.missing_evidence,
        "bad_assessment_errors": bad_report.unsupported_claims + bad_report.missing_evidence
    })


def run_ex8_cost():
    print("--- Running Ex8: Cost & Caching ---")
    pricing = TokenPricing(input_price_per_m=1.0, output_price_per_m=2.0, cached_input_discount=0.5, batch_discount=0.5)
    results = generate_cost_comparison_table(pricing)
    
    print_table("Cost Comparison (Cache & Batch)", results, ["N", "no_cache", "cache", "cache_batch", "savings_pct"])
    write_result("ex8_cost.json", {"comparisons": results})


def run_ex9_schema():
    print("--- Running Ex9: Schema Validation ---")
    
    raw_invalid = """{ "candidate_name": "Jane Doe", "skills": [], "score": 150, "experience_years": 4 }"""
    res = validate_json_output(raw_invalid)
    
    table = [
        {"scenario": "Invalid Score & Empty Skills", "success": res.success, "error_type": res.error_type}
    ]
    print_table("Validation Outcomes", table, ["scenario", "success", "error_type"])
    
    # Ensure errors are JSON serializable
    safe_errors = []
    if res.errors:
        for err in res.errors:
            safe_err = {
                "loc": err.get("loc", []),
                "msg": err.get("msg", ""),
                "type": err.get("type", "")
            }
            safe_errors.append(safe_err)
            
    write_result("ex9_schema.json", {
        "success": res.success,
        "errors": safe_errors,
    })


def main():
    ensure_results_dir()
    
    run_ex1_tokens()
    run_ex2_sampling()
    run_ex3_attention()
    run_ex4_context()
    run_ex5_training()
    run_ex6_tiers()
    run_ex7_hallucination()
    run_ex8_cost()
    run_ex9_schema()
    
    print("ALL EXERCISES COMPLETED.")


if __name__ == "__main__":
    main()
