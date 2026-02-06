"""
Helper functions for references.
"""


def collect_combinations(
    references: dict[str, dict[str, list[str]]],
) -> list[tuple[str, str, list[str]]]:
    """
    Collects combinations of models, datasets and metrics

    Args:
        references (dict[str, dict[str, list[str]]]): references of combinations

    Returns:
        list[tuple[str, str, list[str]]]: list of combinations
    """
    combinations = []
    for model_name, datasets in sorted(references.items()):
        for dataset_name, metrics in sorted(datasets.items()):
            combinations.append((model_name, dataset_name, list(metrics.keys())))
    return combinations


def prepare_result_section(
    results: dict[str, dict[str, dict]], model_name: str, dataset_name: str, metrics: list[str]
) -> None:
    """
    Fill results section with combination.

    Args:
        results (dict[str, dict[str, dict]]): dictionary with results
        model_name (str): model name
        dataset_name (str): dataset name
        metrics (list[str]): metric names
    """
    if model_name not in results:
        results[model_name] = {}
    if dataset_name not in results[model_name]:
        results[model_name][dataset_name] = {}
    for metric in metrics:
        if metric not in results[model_name][dataset_name]:
            results[model_name][dataset_name][metric] = 0.0


def get_generation_models() -> tuple[str, ...]:
    """
    Gets generation models.

    Returns:
        tuple[str, ...]: list of generation models
    """
    return ("VMware/electra-small-mrqa", "timpal0l/mdeberta-v3-base-squad2")


def get_classification_models() -> tuple[str, ...]:
    """
    Gets classification models.

    Returns:
        tuple[str, ...]: list of classification models
    """
    return (
        "cointegrated/rubert-tiny-toxicity",
        "cointegrated/rubert-tiny2-cedr-emotion-detection",
        "papluca/xlm-roberta-base-language-detection",
        "fabriceyhc/bert-base-uncased-ag_news",
        "XSY/albert-base-v2-imdb-calssification",
        "aiknowyou/it-emotion-analyzer",
        "blanchefort/rubert-base-cased-sentiment-rusentiment",
        "tatiana-merz/turkic-cyrillic-classifier",
        "s-nlp/russian_toxicity_classifier",
        "IlyaGusev/rubertconv_toxic_clf",
    )


def get_summurization_models() -> tuple[str, ...]:
    """
    Gets summarization models.

    Returns:
        tuple[str, ...]: list of classification models
    """
    return (
        "mrm8488/bert-mini2bert-mini-finetuned-cnn_daily_mail-summarization",
        "nandakishormpai/t5-small-machine-articles-tag-generation",
        "mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization",
        "stevhliu/my_awesome_billsum_model",
        "UrukHan/t5-russian-summarization",
        "dmitry-vorobiev/rubert_ria_headlines",
    )


def get_nli_models() -> tuple[str, ...]:
    """
    Gets NLI models.

    Returns:
        tuple[str, ...]: list of classification models
    """
    return (
        "cointegrated/rubert-base-cased-nli-threeway",
        "cointegrated/rubert-tiny-bilingual-nli",
        "cross-encoder/qnli-distilroberta-base",
        "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    )


def get_nmt_models() -> tuple[str, ...]:
    """
    Gets NMT models.

    Returns:
        tuple[str, ...]: list of nmt models
    """
    return (
        "Helsinki-NLP/opus-mt-en-fr",
        "Helsinki-NLP/opus-mt-ru-en",
        "Helsinki-NLP/opus-mt-ru-es",
        "google-t5/t5-small",
    )


def get_ner_models() -> tuple[str, ...]:
    """
    Gets NER models.

    Returns:
        tuple[str, ...]: list of ner models
    """
    return (
        "dslim/distilbert-NER",
        "Babelscape/wikineural-multilingual-ner",
    )


def get_open_qa_models() -> tuple[str, ...]:
    """
    Gets Open QA models.

    Returns:
        tuple[str, ...]: list of open qa models
    """
    return (
        "EleutherAI/pythia-160m-deduped",
        "JackFram/llama-68m",
        "EleutherAI/gpt-neo-125m",
    )
