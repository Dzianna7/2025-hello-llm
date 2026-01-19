"""
Checks that the model is being analyzed correctly
"""

import unittest
from pathlib import Path

import pytest
from torch.utils.data.dataset import Dataset

from admin_utils.references.reference_scores import (
    ReferenceAnalysisScores,
    ReferenceAnalysisScoresType,
)
from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.project.lab_settings import LabSettings
from lab_7_llm.main import LLMPipeline


def run_model_analysis_check(lab_path: Path, pipeline_class: type[AbstractLLMPipeline]) -> bool:
    """
    Evaluate metrics from a lab.

    Args:
        lab_path (Path): path to lab
        pipeline_class (type[AbstractLLMPipeline]): pipeline class

    Returns:
        bool: True denotes successful execution.
    """

    settings = LabSettings(lab_path / "settings.json")
    device = "cpu"
    batch_size = 64
    max_length = 120

    fake_dataset = Dataset()
    pipeline = pipeline_class(
        settings.parameters.model, fake_dataset, max_length, batch_size, device
    )
    model_analysis = pipeline.analyze_model()
    references = ReferenceAnalysisScores(scores_type=ReferenceAnalysisScoresType.MODEL)

    model_name = settings.parameters.model.replace("test_", "")

    if references.get(model_name) != model_analysis:
        assert False, f"Model {settings.parameters.model} analysis is incorrect"
    return True


class ModelWorkingTest(unittest.TestCase):
    """
    Tests analyse function
    """

    @pytest.mark.lab_7_llm
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_analyze_ideal(self) -> None:
        """
        Ideal analyze scenario
        """
        self.assertTrue(run_model_analysis_check(Path(__file__).parent.parent, LLMPipeline))
