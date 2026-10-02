"""LLM Client interface with offline MockLLM and optional OpenAICompatLLM backends."""

import json
import os
from abc import ABC, abstractmethod
from typing import Any

import httpx


class LLMClient(ABC):
    """Abstract interface for LLM calls."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate response string from text prompt."""


class MockLLM(LLMClient):
    """Deterministic offline LLM client for testing and local execution."""

    def __init__(self, seed_response: str | None = None):
        self.seed_response = seed_response

    def generate(self, prompt: str, **kwargs: Any) -> str:
        if self.seed_response:
            return self.seed_response

        # Check for specific prompt triggers or return structured assessment default
        prompt_lower = prompt.lower()

        if "does the candidate have kubernetes experience?" in prompt_lower:
            if "sarah chen" in prompt_lower or "kubernetes" in prompt_lower and "techcorp" in prompt_lower:
                return "Based on the provided resume, Sarah Chen does NOT have Kubernetes experience mentioned. Her skills include Docker, AWS (S3, EC2), and CI/CD pipelines."
            elif "marcus vance" in prompt_lower:
                return "Yes, Marcus Vance has extensive Kubernetes experience, managing EKS clusters at CloudScale Systems."
            return "NOT SPECIFIED: The provided resume does not state Kubernetes experience."

        if "lost-in-the-middle key fact probe" in prompt_lower or "key fact" in prompt_lower:
            # Check if key fact is present in prompt context
            if "key fact:" in prompt_lower or "secret key:" in prompt_lower:
                # Extract key fact if found
                for line in prompt.splitlines():
                    if "key fact" in line.lower() or "secret key" in line.lower():
                        return f"Found key fact: {line.strip()}"
                return "Found key fact in context."
            return "Key fact not found in provided text."

        # Default structured JSON response for candidate assessment
        default_assessment = {
            "candidate_name": "Sarah Chen",
            "skills": ["TypeScript", "JavaScript", "React", "Node.js", "PostgreSQL", "Docker", "AWS"],
            "experience_years": 7,
            "score": 85,
            "evidence": [
                "Architected and implemented high-throughput web APIs using Node.js and PostgreSQL",
                "Led frontend design using React and TypeScript, improving page load performance by 35%",
                "7 years of expertise in building scalable web applications"
            ],
            "requirement_assessments": [
                {
                    "requirement": "5+ years of experience with modern web application development",
                    "status": "Meets",
                    "evidence": "7 years of expertise in building scalable web applications"
                },
                {
                    "requirement": "Proficiency in TypeScript, React, and Node.js",
                    "status": "Meets",
                    "evidence": "Architected web APIs using Node.js and React/TypeScript frontend"
                },
                {
                    "requirement": "Experience with Kubernetes",
                    "status": "Does Not Meet",
                    "evidence": "NOT SPECIFIED in candidate resume"
                }
            ],
            "missing_information": ["Direct experience with Kubernetes clusters"],
            "recommendation": "Yes",
            "reasoning": "Candidate strongly meets core requirements for full stack engineering with 7 years of relevant experience in React, Node.js, and PostgreSQL."
        }

        # If custom schema requested in prompt, return custom or valid JSON
        return json.dumps(default_assessment, indent=2)


class OpenAICompatLLM(LLMClient):
    """Optional online backend using OpenAI-compatible API endpoints."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
    ):
        # Read .env manually if exists
        if os.path.exists(".env"):
            with open(".env", "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        if k not in os.environ:
                            os.environ[k.strip()] = v.strip().strip("'").strip('"')
                            
        # If GROQ_API_KEY is found and no specific LLM_API_KEY is set, configure for Groq
        groq_key = os.environ.get("GROQ_API_KEY")
        if groq_key and not os.environ.get("LLM_API_KEY"):
            os.environ["LLM_API_KEY"] = groq_key
            if not os.environ.get("LLM_BASE_URL"):
                os.environ["LLM_BASE_URL"] = "https://api.groq.com/openai/v1"
            if not os.environ.get("LLM_MODEL"):
                # Use a valid Groq model by default
                os.environ["LLM_MODEL"] = "llama-3.1-8b-instant"

        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "OpenAICompatLLM requires an API key passed via init or 'LLM_API_KEY' env variable."
            )

    def generate(self, prompt: str, **kwargs: Any) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": kwargs.get("temperature", 0.0),
        }

        url = f"{self.base_url}/chat/completions"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]


def get_llm_client(backend: str = "mock", **kwargs: Any) -> LLMClient:
    """Factory function for acquiring configured LLMClient backend."""
    if backend == "mock":
        return MockLLM(**kwargs)
    elif backend == "api" or backend == "openaicompat":
        return OpenAICompatLLM(**kwargs)
    else:
        raise ValueError(f"Unknown LLM backend: '{backend}'. Choose 'mock' or 'api'.")
