"""CLI entry points for GenAI Job Application Assistant."""

import argparse
import json
import os
import sys

from jobassist.cost import ProcessingJob, TokenPricing, estimate_cost
from jobassist.hallucination import verify_grounding
from jobassist.llm import get_llm_client
from jobassist.prompts import build_analysis_prompt, build_question_prompt
from jobassist.schema import analyze_with_retry


def read_file(path: str) -> str:
    if not os.path.exists(path):
        print(f"Error: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def cmd_analyze(args):
    """Analyze a resume against a job description."""
    resume_text = read_file(args.resume)
    jd_text = read_file(args.jd)
    
    llm = get_llm_client(backend=args.backend)
    
    # 1. Estimate Cost
    pricing = TokenPricing(input_price_per_m=0.15, output_price_per_m=0.60)
    # Rough estimate of tokens
    static_prefix = 500
    variable_input = len(jd_text.split()) + len(resume_text.split()) # Approx 1 word = 1 token
    job = ProcessingJob(1, static_prefix, variable_input, 250, use_caching=False)
    cost = estimate_cost(job, pricing)
    
    print(f"--- Estimated Cost (Small Tier) ---\n${cost['total_cost']:.6f}\n")
    
    # 2. Analyze with LLM
    print("--- Running Analysis (with Retries) ---")
    prompt = build_analysis_prompt(jd_text, resume_text, include_few_shot=True)
    
    result = analyze_with_retry(llm, prompt, max_retries=2)
    
    if not result.success:
        print("Analysis Failed Validation!")
        print(f"Error Type: {result.error_type}")
        print(f"Details: {result.errors or result.raw_error_message}")
        sys.exit(1)
        
    print("Validated JSON Assessment:")
    output_json = result.data.model_dump_json(indent=2)
    print(output_json)
    print()
    
    # 3. Grounding Verification
    print("--- Grounding Report ---")
    assessment_dict = json.loads(output_json)
    report = verify_grounding(assessment_dict, resume_text)
    
    print(f"Passed Grounding Checks: {report.passed}")
    if report.unsupported_claims:
        print("Unsupported Claims:", report.unsupported_claims)
    if report.missing_evidence:
        print("Missing Evidence:", report.missing_evidence)


def cmd_ask(args):
    """Ask a question based on resume and JD."""
    resume_text = read_file(args.resume)
    jd_text = read_file(args.jd)
    
    llm = get_llm_client(backend=args.backend)
    
    prompt = build_question_prompt(jd_text, resume_text, args.q)
    print("--- Answer ---")
    answer = llm.generate(prompt)
    print(answer)


def cmd_run_exercises(args):
    """Run the 9 exercises via scripts/run_all_exercises.py."""
    # Find the script relative to package or cwd
    script_path = "scripts/run_all_exercises.py"
    if not os.path.exists(script_path):
        print(f"Cannot find {script_path}. Ensure you are in the repository root.")
        sys.exit(1)
        
    # Execute the script
    os.system(f"python {script_path}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        
    parser = argparse.ArgumentParser(description="GenAI Job Application Assistant")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # analyze command
    parser_analyze = subparsers.add_parser("analyze")
    parser_analyze.add_argument("--resume", required=True, help="Path to resume text file")
    parser_analyze.add_argument("--jd", required=True, help="Path to JD text file")
    parser_analyze.add_argument("--backend", default="mock", choices=["mock", "api"], help="LLM backend to use")

    # ask command
    parser_ask = subparsers.add_parser("ask")
    parser_ask.add_argument("--resume", required=True, help="Path to resume text file")
    parser_ask.add_argument("--jd", required=True, help="Path to JD text file")
    parser_ask.add_argument("--q", required=True, help="Question to ask")
    parser_ask.add_argument("--backend", default="mock", choices=["mock", "api"], help="LLM backend to use")

    # run-exercises command
    subparsers.add_parser("run-exercises")

    args = parser.parse_args()

    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "ask":
        cmd_ask(args)
    elif args.command == "run-exercises":
        cmd_run_exercises(args)


if __name__ == "__main__":
    main()
