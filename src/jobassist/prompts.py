"""Prompt templates with cacheable static prefixes and XML structure."""

FEW_SHOT_EXAMPLE = """
Example Assessment Output:
{
  "candidate_name": "Jane Doe",
  "skills": ["Python", "Docker", "SQL"],
  "experience_years": 4,
  "score": 80,
  "evidence": [
    "4 years of experience developing Python web applications",
    "Containerized microservices with Docker"
  ],
  "requirement_assessments": [
    {
      "requirement": "3+ years Python experience",
      "status": "Meets",
      "evidence": "4 years of experience developing Python web applications"
    }
  ],
  "missing_information": [],
  "recommendation": "Yes",
  "reasoning": "Candidate fulfills all primary technical requirements."
}
"""

STATIC_SYSTEM_PREFIX = """<system_role>
You are an expert AI recruiting assistant. Your task is to analyze candidate resumes against job descriptions accurately, objectively, and without hallucination.
</system_role>

<task>
Evaluate the candidate resume against the provided job description and produce a validated JSON assessment object conforming strictly to the requested schema.
</task>

<instructions>
1. Extract the candidate's full name, stated skills, and total years of relevant experience.
2. Calculate a fit score between 0 and 100 based on how well the candidate matches the job description requirements.
3. For every claim made in the assessment, cite exact quotes from the resume in the "evidence" list.
4. Evaluate each key job requirement. If a requirement is not mentioned in the resume, mark its status as "Does Not Meet" or "NOT SPECIFIED" and list it under missing_information. Do NOT assume or hallucinate skills.
5. Provide a recommendation ("Strong Yes", "Yes", "Maybe", "No", "Strong No") consistent with the fit score:
   - Score >= 85: Strong Yes or Yes
   - Score 70-84: Yes or Maybe
   - Score 50-69: Maybe or No
   - Score < 50: No or Strong No
6. Output ONLY valid raw JSON conforming to the schema. Do NOT include commentary outside the JSON block.
</instructions>

<output_schema>
{
  "candidate_name": "string",
  "skills": ["string"],
  "experience_years": "integer (0-50)",
  "score": "integer (0-100)",
  "evidence": ["string (at least 10 chars each)"],
  "requirement_assessments": [
    {
      "requirement": "string",
      "status": "string (Meets / Does Not Meet / NOT SPECIFIED)",
      "evidence": "string"
    }
  ],
  "missing_information": ["string"],
  "recommendation": "string (Strong Yes | Yes | Maybe | No | Strong No)",
  "reasoning": "string"
}
</output_schema>
"""


def build_analysis_prompt(job_description: str, resume: str, include_few_shot: bool = False) -> str:
    """Build prompt with static cacheable prefix first and variable JD/resume last."""
    few_shot = f"\n<few_shot>\n{FEW_SHOT_EXAMPLE.strip()}\n</few_shot>\n" if include_few_shot else ""
    return f"""{STATIC_SYSTEM_PREFIX}{few_shot}
<job_description>
{job_description.strip()}
</job_description>

<resume>
{resume.strip()}
</resume>

<assessment>
"""


def build_question_prompt(job_description: str, resume: str, question: str) -> str:
    """Build prompt for question answering grounded in resume and JD."""
    return f"""<system_role>
You are an AI assistant answering questions about a candidate's resume and job description.
</system_role>

<instructions>
1. Answer the question relying ONLY on the facts explicitly stated in the resume and job description.
2. If the information is not present in the text, explicitly state "NOT SPECIFIED in candidate resume or job description".
3. Do NOT make assumptions or hallucinate information.
</instructions>

<job_description>
{job_description.strip()}
</job_description>

<resume>
{resume.strip()}
</resume>

<question>
{question.strip()}
</question>

<answer>
"""
