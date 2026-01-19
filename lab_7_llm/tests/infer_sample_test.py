"""
Checks the inference of the model
"""

# pylint: disable=R0801
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


def run_model_inference_check(lab_path: Path, pipeline_class: type[AbstractLLMPipeline]) -> bool:
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
    references = ReferenceAnalysisScores(scores_type=ReferenceAnalysisScoresType.INFERENCE)
    model_name = settings.parameters.model.replace("test_", "")
    for query, prediction in references.get(model_name).items():
        if "[TEST SEP]" in query:
            first_value, second_value = query.split("[TEST SEP]")
            res = pipeline.infer_sample((first_value, second_value))
        else:
            res = pipeline.infer_sample((query,))
        if res != prediction:
            assert False, (
                f"Inference of {settings.parameters.model} model is incorrect.\n"
                f"Expected: {prediction}\n"
                f"Actual: {res}"
            )
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
        Ideal inference scenario
        """
        self.assertTrue(run_model_inference_check(Path(__file__).parent.parent, LLMPipeline))
