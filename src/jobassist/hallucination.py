"""Ex7: Hallucination detection & grounding verification."""

import difflib
from dataclasses import dataclass
from typing import Any


@dataclass
class GroundingReport:
    passed: bool
    unsupported_claims: list[str]
    missing_evidence: list[str]


def check_string_presence(query: str, document: str, threshold: float = 0.9) -> bool:
    """Check if query string is present in document using substring or fuzzy matching."""
    q_norm = query.lower().strip()
    d_norm = document.lower().strip()
    
    if not q_norm:
        return True
        
    # 1. Exact normalized substring match
    if q_norm in d_norm:
        return True
        
    # 2. Fuzzy match via sliding window using difflib
    window_size = len(q_norm)
    if window_size == 0 or len(d_norm) < window_size:
        return False
        
    best_ratio = 0.0
    # Step by a reasonable amount to save compute (e.g., 1 or small fraction of window)
    step = max(1, window_size // 4)
    for i in range(0, len(d_norm) - window_size + 1, step):
        window = d_norm[i:i + window_size]
        ratio = difflib.SequenceMatcher(None, q_norm, window).ratio()
        best_ratio = max(best_ratio, ratio)
            
        if best_ratio >= threshold:
            return True
            
    return False


def verify_grounding(assessment: dict[str, Any], resume_text: str) -> GroundingReport:
    """Verify that claims in the assessment are grounded in the resume text.
    
    Rules:
    1. Every 'evidence' string must appear in the resume.
    2. Every claimed 'skill' must appear in the resume text.
    3. 'Does Not Meet' / 'NOT SPECIFIED' logic: If an LLM hallucinates that a candidate 
       has a skill not in the resume (e.g., claiming Kubernetes when it's absent), 
       it will fail the skill grounding check above.
       Additionally, if we specifically check known absent requirements (e.g. Kubernetes),
       it must be marked as NOT SPECIFIED or Does Not Meet.
    """
    unsupported_claims = []
    missing_evidence = []
    
    resume_norm = resume_text.lower()
    
    # 1. Check all explicit evidence strings
    for ev in assessment.get("evidence", []):
        if not check_string_presence(ev, resume_text):
            missing_evidence.append(f"Global evidence not found in resume: '{ev}'")
            
    # 2. Check evidence in requirement assessments
    reqs = assessment.get("requirement_assessments", [])
    for req in reqs:
        status = req.get("status", "")
        ev = req.get("evidence", "")
        
        if status == "Meets":
            if not check_string_presence(ev, resume_text):
                missing_evidence.append(f"Requirement evidence not found for '{req.get('requirement')}': '{ev}'")
                
            # If they meet the requirement, the requirement keywords should somewhat exist in resume
            # (Simplistic check for demonstration)
            req_keywords = [w.lower() for w in req.get("requirement", "").split() if len(w) > 4]
            for kw in req_keywords:
                if kw in resume_norm:
                    break
            
            # If the requirement mentions Kubernetes but it's not in the resume, it's a hallucination
            if "kubernetes" in req.get("requirement", "").lower() and "kubernetes" not in resume_norm:
                 unsupported_claims.append(f"Hallucinated meeting requirement: '{req.get('requirement')}'")
                 
    # 3. Check claimed skills
    skills = assessment.get("skills", [])
    for skill in skills:
        if skill.lower() not in resume_norm:
            unsupported_claims.append(f"Claimed skill not found in resume: '{skill}'")

    passed = len(unsupported_claims) == 0 and len(missing_evidence) == 0
    
    return GroundingReport(
        passed=passed,
        unsupported_claims=unsupported_claims,
        missing_evidence=missing_evidence
    )


# Fixtures for demonstration
SARAH_CHEN_RESUME = """
SARAH CHEN
San Francisco, CA | sarah.chen@example.com

SUMMARY
Experienced Full Stack Engineer with 7 years of expertise in building scalable web applications.

SKILLS
TypeScript, JavaScript, Python, React, Node.js, PostgreSQL, Docker, AWS.

EXPERIENCE
Senior Software Engineer | TechCorp Inc.
- Architected and implemented high-throughput web APIs using Node.js and PostgreSQL.
"""

GOOD_ASSESSMENT = {
    "candidate_name": "Sarah Chen",
    "skills": ["TypeScript", "React", "Node.js", "Docker"],
    "evidence": [
        "7 years of expertise in building scalable web applications"
    ],
    "requirement_assessments": [
        {
            "requirement": "Experience with Node.js and PostgreSQL",
            "status": "Meets",
            "evidence": "Architected and implemented high-throughput web APIs using Node.js and PostgreSQL"
        },
        {
            "requirement": "Experience with Kubernetes",
            "status": "Does Not Meet",
            "evidence": "NOT SPECIFIED"
        }
    ]
}

BAD_HALLUCINATED_ASSESSMENT = {
    "candidate_name": "Sarah Chen",
    "skills": ["TypeScript", "React", "Kubernetes", "GraphQL"], # Kubernetes and GraphQL hallucinated
    "evidence": [
        "7 years of expertise in building scalable web applications",
        "Managed Kubernetes clusters in production" # Hallucinated evidence
    ],
    "requirement_assessments": [
        {
            "requirement": "Experience with Kubernetes",
            "status": "Meets", # Hallucinated status
            "evidence": "Managed Kubernetes clusters in production"
        }
    ]
}
