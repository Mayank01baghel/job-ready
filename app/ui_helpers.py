"""UI helper functions for Streamlit app."""

import io
from typing import Any

from jobassist.cost import ProcessingJob, TokenPricing, estimate_cost
from jobassist.hallucination import verify_grounding
from jobassist.llm import get_llm_client
from jobassist.prompts import build_analysis_prompt
from jobassist.schema import analyze_with_retry


def extract_text_from_bytes(file_bytes: bytes, file_name: str) -> str:
    """Extract text from PDF, DOCX, or TXT file bytes."""
    name_lower = file_name.lower()
    
    if name_lower.endswith(".pdf"):
        import pypdf
        try:
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            text = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
            full_text = "\n".join(text).strip()
            if not full_text:
                raise ValueError("No text found; please upload a text-based file or paste the text")
            return full_text
        except Exception as e:
            if "No text found" in str(e):
                raise
            raise ValueError(f"Could not read PDF: {e}")
            
    elif name_lower.endswith(".docx"):
        import docx
        try:
            doc = docx.Document(io.BytesIO(file_bytes))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            full_text = text.strip()
            if not full_text:
                raise ValueError("No text found; please upload a text-based file or paste the text")
            return full_text
        except Exception as e:
            raise ValueError(f"Could not read DOCX: {e}")
            
    elif name_lower.endswith(".txt") or not "." in name_lower:
        # Try UTF-8, fallback to Latin-1
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return file_bytes.decode("latin-1")
            except Exception as e:
                raise ValueError(f"Could not decode text file: {e}")
    else:
        raise ValueError(f"Unsupported file format for '{file_name}'")


def run_analysis(
    resume_text: str, 
    jd_text: str, 
    backend: str, 
    tier: Any, 
    extra_docs: str = "",
    temperature: float = 0.2
) -> dict[str, Any]:
    """Run full analysis pipeline and return structured results."""
    
    full_resume_text = resume_text
    if extra_docs:
        full_resume_text += "\n\n--- ADDITIONAL DOCUMENTS ---\n\n" + extra_docs
        
    llm = get_llm_client(backend=backend)
    
    # 1. Estimate Cost
    # We'll use the tier prices passed in
    pricing = TokenPricing(
        input_price_per_m=tier.input_price_per_m, 
        output_price_per_m=tier.output_price_per_m
    )
    # Token estimation (rough 1 word = 1.3 tokens for English, more for Hindi but we'll use a fast estimate)
    # Or we can use tiktoken
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        static_prefix = 500  # Approx system prompt
        variable_input = len(enc.encode(jd_text + full_resume_text))
    except Exception:
        # Fallback if tiktoken fails
        static_prefix = 500
        variable_input = int(len((jd_text + full_resume_text).split()) * 1.3)
        
    job_no_cache = ProcessingJob(1, static_prefix, variable_input, 300, use_caching=False)
    job_cache = ProcessingJob(1, static_prefix, variable_input, 300, use_caching=True)
    
    cost_no_cache = estimate_cost(job_no_cache, pricing)
    cost_cache = estimate_cost(job_cache, pricing)
    
    cost_info = {
        "input_tokens": static_prefix + variable_input,
        "output_tokens": 300, # Expected
        "cost_no_cache": cost_no_cache["total_cost"],
        "cost_cache": cost_cache["total_cost"]
    }
    
    # 2. Analyze with LLM
    prompt = build_analysis_prompt(jd_text, full_resume_text, include_few_shot=True)
    
    validation_result = analyze_with_retry(llm, prompt, max_retries=2)
    
    if not validation_result.success:
        return {
            "success": False,
            "error": "Validation failed",
            "validation_result": validation_result,
            "cost_info": cost_info
        }
        
    # 3. Grounding Verification
    assessment_dict = validation_result.data.model_dump()
    report = verify_grounding(assessment_dict, full_resume_text)
    
    return {
        "success": True,
        "assessment": validation_result.data,
        "raw_json": validation_result.data.model_dump_json(indent=2),
        "grounding_report": report,
        "cost_info": cost_info
    }
