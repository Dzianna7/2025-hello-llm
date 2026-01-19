"""
Collect and store inference analytics.
"""

# pylint: disable=import-error, duplicate-code, too-many-branches, no-else-return, inconsistent-return-statements, too-many-locals, too-many-statements, wrong-import-order, too-many-return-statements
from pathlib import Path

from pandas import DataFrame
from pydantic.dataclasses import dataclass
from tqdm import tqdm

from admin_utils.constants import DEVICE
from admin_utils.references.get_model_analytics import get_references, save_reference
from admin_utils.references.helpers import (
    get_classification_models,
    get_generation_models,
    get_ner_models,
    get_nli_models,
    get_nmt_models,
    get_open_qa_models,
    get_summurization_models,
)

from lab_7_llm.main import LLMPipeline, TaskDataset  # isort:skip
from reference_lab_classification.main import ClassificationLLMPipeline  # isort:skip
from reference_lab_generation.main import GenerationLLMPipeline  # isort:skip
from reference_lab_ner.main import NERLLMPipeline  # isort:skip
from reference_lab_nli.main import NLILLMPipeline  # isort:skip
from reference_lab_open_qa.main import OpenQALLMPipeline  # isort:skip


@dataclass
class InferenceParams:
    """
    Inference parameters.
    """

    num_samples: int
    max_length: int
    batch_size: int
    predictions_path: Path
    device: str


def get_inference_from_task(
    model_name: str, inference_params: InferenceParams, samples: list, task: str
) -> dict:
    """
    Gets inferences.

    Args:
        model_name (str): Model name
        inference_params (InferenceParams): Parameters from inference
        samples (list): Samples for inference
        task (str): Task for inference

    Returns:
        dict: Processed predictions with queries
    """
    dataset = TaskDataset(DataFrame([]))

    pipeline_per_task = {
        "nmt": LLMPipeline,
        "generation": GenerationLLMPipeline,
        "classification": ClassificationLLMPipeline,
        "nli": NLILLMPipeline,
        "summarization": LLMPipeline,
        "open_qa": OpenQALLMPipeline,
        "ner": NERLLMPipeline,
    }
    pipeline = pipeline_per_task[task](
        model_name,
        dataset,
        inference_params.max_length,
        inference_params.batch_size,
        inference_params.device,
    )

    result = {}
    for sample in sorted(samples):
        if "[TEST SEP]" in sample:
            first_value, second_value = sample.split("[TEST SEP]")
            prediction = pipeline.infer_sample((first_value, second_value))
        else:
            prediction = pipeline.infer_sample((sample,))

        result[sample] = prediction

    return result


def get_task(model: str, inference_params: InferenceParams, samples: list) -> dict:
    """
    Gets task.

    Args:
        model (str): name of model
        inference_params (InferenceParams): Parameters from inference
        samples (list): Samples for inference

    Returns:
        dict: Results with model predictions
    """
    if "test_" in model:
        model = model.replace("test_", "")

    nmt_model = get_nmt_models()

    generation_model = get_generation_models()

    classification_model = get_classification_models()

    nli_model = get_nli_models()

    summarization_model = get_summurization_models()

    open_qa_model = get_open_qa_models()

    ner_model = get_ner_models()

    if model in nmt_model:
        task = "nmt"
    elif model in generation_model:
        task = "generation"
    elif model in classification_model:
        task = "classification"
    elif model in nli_model:
        task = "nli"
    elif model in summarization_model:
        task = "summarization"
    elif model in open_qa_model:
        task = "open_qa"
    elif model in ner_model:
        task = "ner"
    else:
        raise ValueError(f"Unsupported model {model}")
    return get_inference_from_task(model, inference_params, samples, task)


def main() -> None:
    """
    Run collected reference scores.
    """
    references_dir = Path(__file__).parent / "gold"
    references_path = references_dir / "reference_inference_analytics.json"

    max_length = 120
    batch_size = 1
    num_samples = 100
    device = DEVICE

    inference_params = InferenceParams(
        num_samples=num_samples,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        predictions_path=Path(),
    )

    references = get_references(path=references_path)
    result = {}

    for model in tqdm(sorted(references)):
        print(model, flush=True)
        predictions = get_task(model, inference_params, references[model])
        result[model] = predictions

    save_reference(references_path, result)


if __name__ == "__main__":
    main()
