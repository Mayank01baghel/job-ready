"""Ex9: Structured output validation & Pydantic models."""

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class RequirementAssessment(BaseModel):
    requirement: str
    status: Literal["Meets", "Does Not Meet", "NOT SPECIFIED"]
    evidence: str


class CandidateAssessment(BaseModel):
    candidate_name: str
    skills: list[str]
    experience_years: int = Field(ge=0, le=50)
    score: int = Field(ge=0, le=100)
    evidence: list[str]
    requirement_assessments: list[RequirementAssessment]
    missing_information: list[str]
    recommendation: Literal["Strong Yes", "Yes", "Maybe", "No", "Strong No"]
    reasoning: str

    @field_validator("skills")
    @classmethod
    def check_non_empty_skills(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("Skills list cannot be empty.")
        return v

    @field_validator("evidence")
    @classmethod
    def check_evidence_length(cls, v: list[str]) -> list[str]:
        for idx, item in enumerate(v):
            if len(item.strip()) < 10:
                raise ValueError(f"Evidence item at index {idx} must be at least 10 characters long.")
        return v

    @model_validator(mode="after")
    def check_score_recommendation_consistency(self) -> "CandidateAssessment":
        score = self.score
        rec = self.recommendation

        if score < 30 and rec in ("Strong Yes", "Yes"):
            raise ValueError("Score < 30 cannot result in a 'Yes' or 'Strong Yes' recommendation.")
        if score >= 85 and rec in ("No", "Strong No"):
            raise ValueError("Score >= 85 cannot result in a 'No' or 'Strong No' recommendation.")
            
        return self


class ValidationResult(BaseModel):
    success: bool
    data: CandidateAssessment | None = None
    error_type: str | None = None  # "malformed_json" or "schema_validation"
    errors: list[dict[str, Any]] | None = None
    raw_error_message: str | None = None


def strip_markdown_fences(raw_text: str) -> str:
    """Strip ```json and ``` fences from LLM output."""
    text = raw_text.strip()
    # Remove leading ```json or ```
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    # Remove trailing ```
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def validate_json_output(raw_output: str) -> ValidationResult:
    """Parse, strip markdown, and validate against Pydantic schema."""
    cleaned = strip_markdown_fences(raw_output)
    
    try:
        data_dict = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return ValidationResult(
            success=False,
            error_type="malformed_json",
            raw_error_message=str(e)
        )
        
    try:
        validated_data = CandidateAssessment.model_validate(data_dict)
        return ValidationResult(
            success=True,
            data=validated_data
        )
    except ValidationError as e:
        # Pydantic v2 errors
        errors = e.errors()
        return ValidationResult(
            success=False,
            error_type="schema_validation",
            errors=errors,
            raw_error_message=str(e)
        )


def analyze_with_retry(
    llm_client: Any, 
    prompt: str, 
    max_retries: int = 2
) -> ValidationResult:
    """Execute LLM generation with automated retry loops on validation failure."""
    current_prompt = prompt
    
    for attempt in range(max_retries + 1):
        raw_response = llm_client.generate(current_prompt)
        result = validate_json_output(raw_response)
        
        if result.success:
            return result
            
        # Prepare error feedback for retry
        if attempt < max_retries:
            error_details = result.raw_error_message
            if result.errors:
                error_details = json.dumps(result.errors, indent=2)
                
            feedback_msg = (
                f"\n\nSystem: Your previous response failed validation.\n"
                f"Error type: {result.error_type}\n"
                f"Details:\n{error_details}\n"
                f"Please fix the errors and provide a valid JSON object matching the exact schema."
            )
            current_prompt += feedback_msg
            
    # Return the final failure
    return result
