"""Tests for Exercise 5: Training stages."""

import pytest

from jobassist.training_stages import TrainingStage, explain, identify_training_stage


def test_identify_training_stage():
    assert identify_training_stage("Next token prediction") == TrainingStage.PRETRAINING
    assert identify_training_stage("factual knowledge") == TrainingStage.PRETRAINING
    
    assert identify_training_stage("instruction following") == TrainingStage.SUPERVISED_FINE_TUNING
    assert identify_training_stage("json formatting") == TrainingStage.SUPERVISED_FINE_TUNING
    
    assert identify_training_stage("avoid hallucination") == TrainingStage.PREFERENCE_TUNING
    assert identify_training_stage("tone alignment") == TrainingStage.PREFERENCE_TUNING
    
    with pytest.raises(ValueError):
        identify_training_stage("unknown random behavior")


def test_explain_training_stage():
    sft = explain(TrainingStage.SUPERVISED_FINE_TUNING)
    assert "data" in sft
    assert "objective" in sft
    assert "example_io" in sft
    
    rlhf = explain(TrainingStage.PREFERENCE_TUNING)
    assert "nuance" in rlhf
    assert "hallucination" in rlhf["nuance"]
