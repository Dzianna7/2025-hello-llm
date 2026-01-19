"""
Collect and store model analytics.
"""

# pylint: disable=import-error, too-many-branches, no-else-return, inconsistent-return-statements, too-many-locals, too-many-statements, wrong-import-order, too-many-return-statements
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
from typing import Any

from pydantic.dataclasses import dataclass
from tqdm import tqdm

from admin_utils.constants import DEVICE
from admin_utils.references.get_model_analytics import get_references, save_reference
from admin_utils.references.helpers import (
    collect_combinations,
    get_classification_models,
    get_generation_models,
    get_ner_models,
    get_nli_models,
    get_nmt_models,
    get_open_qa_models,
    get_summurization_models,
    prepare_result_section,
)
from core_utils.llm.metrics import Metrics
from core_utils.project.lab_settings import InferenceParams

from reference_lab_classification.start import get_result_for_classification  # isort:skip
from reference_lab_generation.start import get_result_for_generation  # isort:skip
from reference_lab_ner.start import get_result_for_ner  # isort:skip
from reference_lab_nli.start import get_result_for_nli  # isort:skip
from reference_lab_nmt.start import get_result_for_nmt  # isort:skip
from reference_lab_open_qa.start import get_result_for_open_qa  # isort:skip
from reference_lab_summarization.start import get_result_for_summarization  # isort:skip


@dataclass
class MainParams:
    """
    Main parameters.
    """

    model: str
    dataset: str
    metrics: list[Metrics]


def get_task(model: str, main_params: MainParams, inference_params: InferenceParams) -> Any:
    """
    Gets task.

    Args:
        model (str): name of model
        main_params (MainParams): Parameters from main
        inference_params (InferenceParams): Parameters from inference

    Returns:
        Any: Metric for a specific task
    """
    if "test_" in model:
        model = model.replace("test_", "")

    nmt_model = get_nmt_models()
    generation_model = get_generation_models()
    classification_models = get_classification_models()
    nli_model = get_nli_models()
    summarization_model = get_summurization_models()
    open_generative_qa_model = get_open_qa_models()
    ner_model = get_ner_models()

    if model in nmt_model:
        return get_result_for_nmt(inference_params, main_params)
    elif model in generation_model:
        return get_result_for_generation(inference_params, main_params)
    elif model in classification_models:
        return get_result_for_classification(inference_params, main_params)
    elif model in nli_model:
        return get_result_for_nli(inference_params, main_params)
    elif model in summarization_model:
        return get_result_for_summarization(inference_params, main_params)
    elif model in open_generative_qa_model:
        return get_result_for_open_qa(inference_params, main_params)
    elif model in ner_model:
        return get_result_for_ner(inference_params, main_params)
    else:
        raise ValueError(f"Unknown model {model} ...")


def main() -> None:
    """
    Run collected reference scores.
    """
    project_root = Path(__file__).parent.parent.parent
    references_dir = project_root / "admin_utils" / "references" / "gold"
    references_path = references_dir / "reference_scores.json"

    dist_dir = project_root / "dist"
    dist_dir.mkdir(exist_ok=True)

    max_length = 120
    batch_size = 3
    num_samples = 100
    device = DEVICE

    inference_params = InferenceParams(
        num_samples, max_length, batch_size, dist_dir / "result.csv", device
    )

    references = get_references(path=references_path)
    combos = collect_combinations(references)

    result = {}
    elements = list(enumerate(sorted(combos)))
    for _, (model_name, dataset_name, metrics) in tqdm(elements, total=len(elements)):
        print(model_name, dataset_name, metrics, flush=True)

        prepare_result_section(result, model_name, dataset_name, metrics)

        if "test_" in model_name:
            inference_params.num_samples = 10

        main_params = MainParams(model_name, dataset_name, [Metrics(metric) for metric in metrics])
        inference_result = get_task(model_name, main_params, inference_params)
        for metric in metrics:
            score = Decimal(inference_result[metric]).quantize(Decimal("1.00000"), ROUND_FLOOR)
            result[model_name][dataset_name][metric] = score
    save_reference(references_path, result)


if __name__ == "__main__":
    main()
