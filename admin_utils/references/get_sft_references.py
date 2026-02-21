"""
Collect and store model analytics.
"""

# pylint: disable=import-error, wrong-import-order, duplicate-code, too-many-locals
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
from typing import Any

from pydantic.dataclasses import dataclass
from tqdm import tqdm

try:
    from transformers import set_seed
except ImportError:
    print('Library "transformers" not installed. Failed to import.')

from admin_utils.constants import (
    DEVICE,
    GLOBAL_FINE_TUNING_BATCH_SIZE,
    GLOBAL_INFERENCE_BATCH_SIZE,
    GLOBAL_MAX_LENGTH,
    GLOBAL_NUM_SAMPLES,
    GLOBAL_SEED,
)
from admin_utils.references.get_model_analytics import get_references, save_reference
from admin_utils.references.helpers import (
    collect_combinations,
    get_classification_models,
    get_ner_models,
    get_nli_models,
    get_nmt_models,
    get_summurization_models,
    prepare_result_section,
)
from core_utils.llm.metrics import Metrics
from core_utils.project.lab_settings import InferenceParams, SFTParams

from reference_lab_classification_sft.start import get_result_for_classification  # isort:skip
from reference_lab_nli_sft.start import get_result_for_nli  # isort:skip
from reference_lab_ner_sft.start import get_result_for_ner  # isort:skip
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


def get_target_modules(  # pylint: disable=too-many-return-statements)
    model_name: str,
) -> list[str] | None:
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
    if model_name in ("tatiana-merz/turkic-cyrillic-classifier",):
        return ["query"]
    if model_name in ("cointegrated/rubert-base-cased-nli-threeway"):
        return ["query"]
    if model_name in (
        "Helsinki-NLP/opus-mt-ru-en",
        "Helsinki-NLP/opus-mt-ru-es",
        "Helsinki-NLP/opus-mt-en-fr",
    ):
        return ["q_proj", "k_proj"]
    if model_name in ("stevhliu/my_awesome_billsum_model",):
        return ["q"]
    if model_name in ("google-t5/t5-small",):
        return ["q", "k", "v"]
    if model_name in ("UrukHan/t5-russian-summarization",):
        return ["q", "k", "wi", "wo"]
    if model_name in ("dslim/distilbert-NER",):
        return ["q_lin", "k_lin", "v_lin", "out_lin"]
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
    ner_model = get_ner_models()

    if model in classification_models:
        fine_tuning_pipeline = get_result_for_classification
    elif model in summarization_models:
        fine_tuning_pipeline = get_result_for_summarization
    elif model in nli_models:
        fine_tuning_pipeline = get_result_for_nli
    elif model in nmt_models:
        fine_tuning_pipeline = get_result_for_nmt
    elif model in ner_model:
        fine_tuning_pipeline = get_result_for_ner
    else:
        raise ValueError(f"Unknown model {model} ...")
    return fine_tuning_pipeline(inference_params, sft_params, main_params)


def main() -> None:
    """
    Run collected reference scores.
    """
    set_seed(GLOBAL_SEED)

    project_root = Path(__file__).parent.parent.parent
    references_path = (
        project_root / "admin_utils" / "references" / "gold" / "reference_sft_scores.json"
    )

    dist_dir = project_root / "dist"
    dist_dir.mkdir(exist_ok=True)

    references = get_references(path=references_path)

    combinations = collect_combinations(references)

    inference_params = InferenceParams(
        num_samples=GLOBAL_NUM_SAMPLES,
        max_length=GLOBAL_MAX_LENGTH,
        batch_size=GLOBAL_INFERENCE_BATCH_SIZE,
        predictions_path=dist_dir / "predictions.csv",
        device=DEVICE,
    )

    sft_params = SFTParams(
        batch_size=GLOBAL_FINE_TUNING_BATCH_SIZE,
        finetuned_model_path=dist_dir,
        device=DEVICE,
        max_length=GLOBAL_MAX_LENGTH,
        learning_rate=1e-3,
        max_fine_tuning_steps=50,
        rank=8,
        alpha=8,
        target_modules=None,
    )

    specific_fine_tuning_steps = {
        "cointegrated/rubert-base-cased-nli-threeway": 100,
        "Helsinki-NLP/opus-mt-en-fr": 60,
        "Helsinki-NLP/opus-mt-ru-es": 100,
        "mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization": 150,
    }

    specific_rank = {
        "Helsinki-NLP/opus-mt-en-fr": 8,
        "cointegrated/rubert-tiny2-cedr-emotion-detection": 16,
        "mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization": 24,
        "google-t5/t5-small": 24,
    }
    specific_alpha = {
        "Helsinki-NLP/opus-mt-en-fr": 8,
        "cointegrated/rubert-tiny2-cedr-emotion-detection": 24,
        "stevhliu/my_awesome_billsum_model": 48,
        "mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization": 36,
        "google-t5/t5-small": 36,
    }

    result = {}
    for model_name, dataset_name, metrics in tqdm(sorted(combinations)):
        print(model_name, dataset_name, metrics, flush=True)
        prepare_result_section(result, model_name, dataset_name, metrics)

        sft_params.finetuned_model_path = dist_dir / model_name
        sft_params.max_fine_tuning_steps = specific_fine_tuning_steps.get(model_name, 50)
        sft_params.rank = specific_rank.get(model_name, 16)

        sft_params.alpha = specific_alpha.get(model_name, 16)
        if (
            model_name == "stevhliu/my_awesome_billsum_model"
            and dataset_name == "CarlBrendt/Summ_Dialog_News"
        ):
            sft_params.alpha = 36
        if (
            model_name == "mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization"
            and dataset_name == "cnn_dailymail"
        ):
            sft_params.alpha = sft_params.rank * 2
            sft_params.learning_rate = 1e-5

        sft_params.target_modules = get_target_modules(model_name)

        main_params = MainParams(model_name, dataset_name, [Metrics(metric) for metric in metrics])

        sft_result = get_task(model_name, main_params, inference_params, sft_params)
        for metric in metrics:
            score = Decimal(sft_result[metric]).quantize(Decimal("1.00000"), ROUND_FLOOR)
            result[model_name][dataset_name][metric] = score
    save_reference(references_path, result)


if __name__ == "__main__":
    main()
