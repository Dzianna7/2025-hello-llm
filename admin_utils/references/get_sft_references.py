"""
Collect and store model analytics.
"""

# pylint: disable=import-error, wrong-import-order, duplicate-code, too-many-locals
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
    get_nli_models,
    get_nmt_models,
    get_summurization_models,
    prepare_result_section,
)
from core_utils.llm.metrics import Metrics
from core_utils.project.lab_settings import InferenceParams, SFTParams

from reference_lab_classification_sft.start import get_result_for_classification  # isort:skip
from reference_lab_nli_sft.start import get_result_for_nli  # isort:skip
from reference_lab_nmt_sft.start import get_result_for_nmt  # isort:skip
from reference_lab_summarization_sft.start import get_result_for_summarization  # isort:skip


@dataclass
class MainParams:
    """
    Main parameters.
    """

    model: str
    dataset: str
    metrics: list[Metrics]


def get_target_modules(model_name: str) -> list[str] | None:
    """
    Gets modules to fine-tune with LoRA.

    Args:
        model_name (str): Model name

    Returns:
        list[str] | None: modules to fine-tune with LoRA.
    """
    if model_name in (
        "dmitry-vorobiev/rubert_ria_headlines",
        "XSY/albert-base-v2-imdb-calssification",
        "mrm8488/bert-mini2bert-mini-finetuned-cnn_daily_mail-summarization",
        "mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization",
        "cointegrated/rubert-tiny2-cedr-emotion-detection",
    ):
        return ["query", "key", "value", "dense"]
    if model_name in ("cointegrated/rubert-base-cased-nli-threeway"):
        return ["key"]
    if model_name in (
        "Helsinki-NLP/opus-mt-ru-en",
        "Helsinki-NLP/opus-mt-ru-es",
        "Helsinki-NLP/opus-mt-en-fr",
    ):
        return ["q_proj", "k_proj"]
    if model_name in ("stevhliu/my_awesome_billsum_model", "google-t5/t5-small"):
        return ["q", "k", "v"]
    if model_name in ("UrukHan/t5-russian-summarization",):
        return ["q", "k", "wi", "wo"]
    return None


def get_task(
    model: str,
    main_params: MainParams,
    inference_params: InferenceParams,
    sft_params: SFTParams,
) -> Any:
    """
    Gets task.

    Args:
        model (str): name of model
        main_params (MainParams): Parameters from main
        inference_params (InferenceParams): Parameters for inference
        sft_params (SFTParams): Parameters for fine-tuning

    Returns:
        Any: Metric for a specific task
    """
    if "test_" in model:
        model = model.replace("test_", "")

    classification_models = get_classification_models()
    summarization_models = get_summurization_models()
    nli_models = get_nli_models()
    nmt_models = get_nmt_models()

    if model in classification_models:
        fine_tuning_pipeline = get_result_for_classification
    elif model in summarization_models:
        fine_tuning_pipeline = get_result_for_summarization
    elif model in nli_models:
        fine_tuning_pipeline = get_result_for_nli
    elif model in nmt_models:
        fine_tuning_pipeline = get_result_for_nmt
    else:
        raise ValueError(f"Unknown model {model} ...")
    return fine_tuning_pipeline(inference_params, sft_params, main_params)


def main() -> None:
    """
    Run collected reference scores.
    """
    project_root = Path(__file__).parent.parent.parent
    references_path = (
        project_root / "admin_utils" / "references" / "gold" / "reference_sft_scores.json"
    )

    dist_dir = project_root / "dist"
    dist_dir.mkdir(exist_ok=True)

    references = get_references(path=references_path)

    combinations = collect_combinations(references)

    inference_params = InferenceParams(
        num_samples=100,
        max_length=120,
        batch_size=64,
        predictions_path=dist_dir / "predictions.csv",
        device=DEVICE,
    )

    sft_params = SFTParams(
        batch_size=3,
        finetuned_model_path=dist_dir,
        device=DEVICE,
        max_length=120,
        learning_rate=1e-3,
        max_fine_tuning_steps=50,
        rank=8,
        alpha=8,
        target_modules=None,
    )

    specific_fine_tuning_steps = {
        "Helsinki-NLP/opus-mt-en-fr": 60,
        "Helsinki-NLP/opus-mt-ru-es": 100,
        "stevhliu/my_awesome_billsum_model": 60,
        "mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization": 150,
    }
    specific_lr = {
        "stevhliu/my_awesome_billsum_model": 1e-4,
    }
    specific_rank = {
        "Helsinki-NLP/opus-mt-en-fr": 16,
        "cointegrated/rubert-tiny2-cedr-emotion-detection": 16,
        "stevhliu/my_awesome_billsum_model": 24,
        "mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization": 24,
        "google-t5/t5-small": 24,
    }
    specific_alpha = {
        "Helsinki-NLP/opus-mt-en-fr": 24,
        "cointegrated/rubert-tiny2-cedr-emotion-detection": 24,
        "stevhliu/my_awesome_billsum_model": 36,
        "mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization": 36,
        "google-t5/t5-small": 36,
    }

    result = {}
    for model_name, dataset_name, metrics in tqdm(sorted(combinations)):
        if (
            model_name == "cointegrated/rubert-tiny-toxicity"
            and dataset_name == "OxAISH-AL-LLM/wiki_toxic"
        ):
            continue
        print(model_name, dataset_name, metrics, flush=True)
        prepare_result_section(result, model_name, dataset_name, metrics)

        sft_params.finetuned_model_path = dist_dir / model_name
        sft_params.learning_rate = specific_lr.get(model_name, 1e-3)
        sft_params.max_fine_tuning_steps = specific_fine_tuning_steps.get(model_name, 50)
        sft_params.rank = specific_rank.get(model_name, 8)
        sft_params.alpha = specific_alpha.get(model_name, 8)
        sft_params.target_modules = get_target_modules(model_name)

        main_params = MainParams(model_name, dataset_name, [Metrics(metric) for metric in metrics])

        sft_result = get_task(model_name, main_params, inference_params, sft_params)
        for metric in metrics:
            score = Decimal(sft_result[metric]).quantize(Decimal("1.00000"), ROUND_FLOOR)
            result[model_name][dataset_name][metric] = score
    save_reference(references_path, result)


if __name__ == "__main__":
    main()
