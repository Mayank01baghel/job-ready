"""Ex5: Training stages."""

from enum import Enum
from typing import Any


class TrainingStage(Enum):
    PRETRAINING = "pretraining"
    SUPERVISED_FINE_TUNING = "supervised_fine_tuning"
    PREFERENCE_TUNING = "preference_tuning"


def identify_training_stage(behavior: str) -> TrainingStage:
    """Identify which training stage is primarily responsible for a given behavior."""
    b_lower = behavior.lower()
    
    pretraining_behaviors = [
        "next token prediction",
        "grammar",
        "factual knowledge",
        "language modeling"
    ]
    
    sft_behaviors = [
        "instruction following",
        "json formatting",
        "q&a formatting",
        "structured output"
    ]
    
    preference_behaviors = [
        "avoid hallucination",
        "refusal of harmful queries",
        "tone alignment",
        "safety"
    ]
    
    for pb in pretraining_behaviors:
        if pb in b_lower:
            return TrainingStage.PRETRAINING
            
    for sb in sft_behaviors:
        if sb in b_lower:
            return TrainingStage.SUPERVISED_FINE_TUNING
            
    for rlhf in preference_behaviors:
        if rlhf in b_lower:
            return TrainingStage.PREFERENCE_TUNING
            
    # FIXED: The original roadmap silently defaulted to PRETRAINING.
    # We explicitly raise ValueError for unknown behaviors to catch bugs.
    raise ValueError(f"Unknown behavior: '{behavior}'. Cannot map to a training stage.")


def explain(stage: TrainingStage) -> dict[str, Any]:
    """Provide an explanation for the given training stage."""
    if stage == TrainingStage.PRETRAINING:
        return {
            "data": "Massive uncurated web scrape (trillions of tokens).",
            "objective": "Next-token prediction (unsupervised/self-supervised).",
            "example_io": {
                "input": "The capital of France is",
                "output": " Paris."
            }
        }
    elif stage == TrainingStage.SUPERVISED_FINE_TUNING:
        return {
            "data": "High-quality human-annotated instruction-response pairs (thousands to millions).",
            "objective": "Behavior cloning (learn to format and respond as an assistant).",
            "example_io": {
                "input": "User: What is the capital of France?\nAssistant:",
                "output": " The capital of France is Paris."
            }
        }
    elif stage == TrainingStage.PREFERENCE_TUNING:
        return {
            "data": "Human or AI preference rankings over multiple model responses.",
            "objective": "Maximize expected reward / align with human preferences (RLHF/DPO).",
            "example_io": {
                "input": "User: Write a harmful script.\nAssistant:",
                "output": " I cannot fulfill this request."
            },
            "nuance": "Behaviors like 'avoid hallucination' are improved by preference tuning but not fully solved by it."
        }
    
    raise ValueError(f"Unknown stage: {stage}")
