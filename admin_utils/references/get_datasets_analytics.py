"""
Collect and store dataset analytics.
"""

# pylint: disable=import-error, too-many-branches, too-many-statements, wrong-import-order, duplicate-code
import sys
from logging import warning
from pathlib import Path

from tqdm import tqdm

try:
    from transformers import set_seed
except ImportError:
    print('Library "transformers" not installed. Failed to import.')

from admin_utils.constants import GLOBAL_SEED
from admin_utils.references.models import (
    DatasetReferenceDTO,
    DatasetReferencesModel,
    EvaluationReferencesModel,
)
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor

from lab_7_llm.main import RawDataImporter, RawDataPreprocessor  # isort:skip
from reference_lab_classification.main import (  # isort:skip
    AgNewsDataImporter,
    AgNewsPreprocessor,
    CyrillicTurkicDataImporter,
    CyrillicTurkicPreprocessor,
    DairAiEmotionDataImporter,
    DairAiEmotionPreprocessor,
    GoEmotionsDataImporter,
    GoEmotionsRawDataPreprocessor,
    HealthcareDataImporter,
    HealthcarePreprocessor,
    ImdbDataImporter,
    ImdbDataPreprocessor,
    KinopoiskDataImporter,
    KinopoiskPreprocessor,
    LanguageIdentificationDataImporter,
    LanguageIdentificationPreprocessor,
    ParadetoxDataImporter,
    ParadetoxDataPreprocessor,
    RuDetoxifierDataImporter,
    RuDetoxifierPreprocessor,
    RuGoEmotionsRawDataPreprocessor,
    RuGoRawDataImporter,
    RuNonDetoxifiedDataImporter,
    RuNonDetoxifiedPreprocessor,
    RuParadetoxDataImporter,
    RuParadetoxPreprocessor,
    ToxicityDataImporter,
    ToxicityDataPreprocessor,
    WikiToxicDataImporter,
    WikiToxicRawDataPreprocessor,
)
from reference_lab_generation.main import (  # isort:skip
    ClinicalNotesRawDataImporter,
    ClinicalNotesRawDataPreprocessor,
    DollyClosedRawDataImporter,
    DollyClosedRawDataPreprocessor,
    NoRobotsRawDataImporter,
    NoRobotsRawDataPreprocessor,
    SberquadRawDataImporter,
    SberquadRawDataPreprocessor,
    WikiOmniaRawDataImporter,
    WikiOmniaRawDataPreprocessor,
)
from reference_lab_ner.main import (  # isort:skip
    Conll2003DataImporter,
    NERRawDataPreprocessor,
    WikineuralDataImporter,
)
from reference_lab_nli.main import (  # isort:skip
    DatasetTypes,
    GlueDataImporter,
    NliDataPreprocessor,
    NliRusDataImporter,
    NliRusTranslatedDataPreprocessor,
    QnliDataPreprocessor,
    RussianSuperGlueDataImporte,
    XnliDataImporter,
)
from reference_lab_nmt.main import (  # isort:skip
    EnDeRawDataPreprocessor,
    RuEnRawDataImporter,
    RuEnRawDataPreprocessor,
    RuEsRawDataPreprocessor,
)
from reference_lab_open_qa.main import (  # isort:skip
    AlpacaRawDataPreprocessor,
    DatabricksRawDataPreprocessor,
    DollyOpenQARawDataImporter,
    DollyOpenQARawDataPreprocessor,
    QARawDataImporter,
    TruthfulQARawDataImporter,
    TruthfulQARawDataPreprocessor,
)
from reference_lab_summarization.main import (  # isort:skip
    DailymailRawDataImporter,
    DailymailRawDataPreprocessor,
    GovReportRawDataPreprocessor,
    PubMedRawDataPreprocessor,
    RuCorpusRawDataImporter,
    RuCorpusRawDataPreprocessor,
    RuDialogNewsRawDataPreprocessor,
    RuGazetaRawDataPreprocessor,
    RuReviewsRawDataImporter,
    RuReviewsRawDataPreprocessor,
    SummarizationRawDataImporter,
)


def main() -> None:
    """
    Run the collect dataset analytics.
    """
    set_seed(GLOBAL_SEED)

    references_dir = Path(__file__).parent / "gold"
    references_path = references_dir / "reference_scores.json"
    destination_path = references_dir / "reference_dataset_analytics.json"

    eval_references = EvaluationReferencesModel.from_json(references_path)
    datasets = eval_references.get_datasets()

    dataset_references = DatasetReferencesModel()

    for dataset_name in tqdm(datasets):
        importer: AbstractRawDataImporter
        print(f"Processing {dataset_name} ...", flush=True)

        if dataset_name == "seara/ru_go_emotions":
            importer = RuGoRawDataImporter(dataset_name)
        elif dataset_name == "imdb":
            importer = ImdbDataImporter(dataset_name)
        elif dataset_name == "dair-ai/emotion":
            importer = DairAiEmotionDataImporter(dataset_name)
        elif dataset_name == "ag_news":
            importer = AgNewsDataImporter(dataset_name)
        elif dataset_name == "papluca/language-identification":
            importer = LanguageIdentificationDataImporter(dataset_name)
        elif dataset_name == "OxAISH-AL-LLM/wiki_toxic":
            importer = WikiToxicDataImporter(dataset_name)
        elif dataset_name == "go_emotions":
            importer = GoEmotionsDataImporter(dataset_name)
        elif dataset_name == "lionelchg/dolly_closed_qa":
            importer = DollyClosedRawDataImporter(dataset_name)
        elif dataset_name == "starmpcc/Asclepius-Synthetic-Clinical-Notes":
            importer = ClinicalNotesRawDataImporter(dataset_name)
        elif dataset_name == "HuggingFaceH4/no_robots":
            importer = NoRobotsRawDataImporter(dataset_name)
        elif dataset_name == "sberquad":
            importer = SberquadRawDataImporter(dataset_name)
        elif dataset_name == "RussianNLP/wikiomnia":
            importer = WikiOmniaRawDataImporter(dataset_name)
        elif dataset_name == DatasetTypes.XNLI.value:
            importer = XnliDataImporter(dataset_name)
        elif dataset_name == DatasetTypes.NLI_RUS.value:
            importer = NliRusDataImporter(dataset_name)
        elif dataset_name == DatasetTypes.MNLI.value:
            importer = GlueDataImporter(dataset_name)
        elif dataset_name == DatasetTypes.QNLI.value:
            importer = GlueDataImporter(dataset_name)
        elif dataset_name == DatasetTypes.TERRA.value:
            importer = RussianSuperGlueDataImporte(dataset_name)
        elif dataset_name == "cnn_dailymail":
            importer = DailymailRawDataImporter(dataset_name)
        elif dataset_name == "d0rj/curation-corpus-ru":
            importer = RuCorpusRawDataImporter(dataset_name)
        elif dataset_name == "trixdade/reviews_russian":
            importer = RuReviewsRawDataImporter(dataset_name)
        elif dataset_name in [
            "ccdv/pubmed-summarization",
            "ccdv/govreport-summarization",
            "IlyaGusev/gazeta",
            "CarlBrendt/Summ_Dialog_News",
        ]:
            importer = SummarizationRawDataImporter(dataset_name)
        elif dataset_name == "shreevigneshs/iwslt-2023-en-ru-train-val-split-0.2":
            importer = RuEnRawDataImporter(dataset_name)
        elif dataset_name == "blinoff/kinopoisk":
            importer = KinopoiskDataImporter(dataset_name)
        elif dataset_name == "blinoff/healthcare_facilities_reviews":
            importer = HealthcareDataImporter(dataset_name)
        elif dataset_name == "tatiana-merz/cyrillic_turkic_langs":
            importer = CyrillicTurkicDataImporter(dataset_name)
        elif dataset_name == "s-nlp/ru_paradetox_toxicity":
            importer = RuParadetoxDataImporter(dataset_name)
        elif dataset_name == "s-nlp/ru_non_detoxified":
            importer = RuNonDetoxifiedDataImporter(dataset_name)
        elif dataset_name == "d0rj/rudetoxifier_data":
            importer = RuDetoxifierDataImporter(dataset_name)
        elif dataset_name == "domenicrosati/TruthfulQA":
            importer = TruthfulQARawDataImporter(dataset_name)
        elif dataset_name in ["tatsu-lab/alpaca", "jtatman/databricks-dolly-8k-qa-open-close"]:
            importer = QARawDataImporter(dataset_name)
        elif dataset_name == "lionelchg/dolly_open_qa":
            importer = DollyOpenQARawDataImporter(dataset_name)
        elif dataset_name == "Arsive/toxicity_classification_jigsaw":
            importer = ToxicityDataImporter(dataset_name)
        elif dataset_name == "s-nlp/en_paradetox_toxicity":
            importer = ParadetoxDataImporter(dataset_name)
        elif dataset_name == "eriktks/conll2003":
            importer = Conll2003DataImporter(dataset_name)
        elif dataset_name == "Babelscape/wikineural":
            importer = WikineuralDataImporter(dataset_name)
        else:
            importer = RawDataImporter(dataset_name)
            warning(f"Using default importer for {dataset_name}")

        importer.obtain()

        if importer.raw_data is None:
            print("Raw data is empty")
            sys.exit(1)
        preprocessor: AbstractRawDataPreprocessor
        if dataset_name == "OxAISH-AL-LLM/wiki_toxic":
            preprocessor = WikiToxicRawDataPreprocessor(importer.raw_data)
        elif dataset_name == "go_emotions":
            preprocessor = GoEmotionsRawDataPreprocessor(importer.raw_data)
        elif dataset_name == "seara/ru_go_emotions":
            preprocessor = RuGoEmotionsRawDataPreprocessor(importer.raw_data)
        elif dataset_name == "imdb":
            preprocessor = ImdbDataPreprocessor(importer.raw_data)
        elif dataset_name == "dair-ai/emotion":
            preprocessor = DairAiEmotionPreprocessor(importer.raw_data)
        elif dataset_name == "ag_news":
            preprocessor = AgNewsPreprocessor(importer.raw_data)
        elif dataset_name == "papluca/language-identification":
            preprocessor = LanguageIdentificationPreprocessor(importer.raw_data)
        elif dataset_name == "lionelchg/dolly_closed_qa":
            preprocessor = DollyClosedRawDataPreprocessor(importer.raw_data)
        elif dataset_name == "starmpcc/Asclepius-Synthetic-Clinical-Notes":
            preprocessor = ClinicalNotesRawDataPreprocessor(importer.raw_data)
        elif dataset_name == "HuggingFaceH4/no_robots":
            preprocessor = NoRobotsRawDataPreprocessor(importer.raw_data)
        elif dataset_name == "sberquad":
            preprocessor = SberquadRawDataPreprocessor(importer.raw_data)
        elif dataset_name == "RussianNLP/wikiomnia":
            preprocessor = WikiOmniaRawDataPreprocessor(importer.raw_data)
        elif dataset_name in (
            DatasetTypes.XNLI.value,
            DatasetTypes.MNLI.value,
            DatasetTypes.TERRA.value,
        ):
            preprocessor = NliDataPreprocessor(importer.raw_data)
        elif dataset_name == DatasetTypes.NLI_RUS.value:
            preprocessor = NliRusTranslatedDataPreprocessor(importer.raw_data)
        elif dataset_name == DatasetTypes.QNLI.value:
            preprocessor = QnliDataPreprocessor(importer.raw_data)
        elif dataset_name == "ccdv/pubmed-summarization":
            preprocessor = PubMedRawDataPreprocessor(importer.raw_data)
        elif dataset_name == "ccdv/govreport-summarization":
            preprocessor = GovReportRawDataPreprocessor(importer.raw_data)
        elif dataset_name == "cnn_dailymail":
            preprocessor = DailymailRawDataPreprocessor(importer.raw_data)
        elif dataset_name == "IlyaGusev/gazeta":
            preprocessor = RuGazetaRawDataPreprocessor(importer.raw_data)
        elif dataset_name == "CarlBrendt/Summ_Dialog_News":
            preprocessor = RuDialogNewsRawDataPreprocessor(importer.raw_data)
        elif dataset_name == "trixdade/reviews_russian":
            preprocessor = RuReviewsRawDataPreprocessor(importer.raw_data)
        elif dataset_name == "d0rj/curation-corpus-ru":
            preprocessor = RuCorpusRawDataPreprocessor(importer.raw_data)
        elif dataset_name == "shreevigneshs/iwslt-2023-en-ru-train-val-split-0.2":
            preprocessor = RuEnRawDataPreprocessor(importer.raw_data)
        elif dataset_name == "nuvocare/Ted2020_en_es_fr_de_it_ca_pl_ru_nl":
            preprocessor = RuEsRawDataPreprocessor(importer.raw_data)
        elif dataset_name == "RocioUrquijo/en_de":
            preprocessor = EnDeRawDataPreprocessor(importer.raw_data)
        elif dataset_name == "blinoff/kinopoisk":
            preprocessor = KinopoiskPreprocessor(importer.raw_data)
        elif dataset_name == "blinoff/healthcare_facilities_reviews":
            preprocessor = HealthcarePreprocessor(importer.raw_data)
        elif dataset_name == "tatiana-merz/cyrillic_turkic_langs":
            preprocessor = CyrillicTurkicPreprocessor(importer.raw_data)
        elif dataset_name == "s-nlp/ru_paradetox_toxicity":
            preprocessor = RuParadetoxPreprocessor(importer.raw_data)
        elif dataset_name == "s-nlp/ru_non_detoxified":
            preprocessor = RuNonDetoxifiedPreprocessor(importer.raw_data)
        elif dataset_name == "d0rj/rudetoxifier_data":
            preprocessor = RuDetoxifierPreprocessor(importer.raw_data)
        elif dataset_name == "domenicrosati/TruthfulQA":
            preprocessor = TruthfulQARawDataPreprocessor(importer.raw_data)
        elif dataset_name == "jtatman/databricks-dolly-8k-qa-open-close":
            preprocessor = DatabricksRawDataPreprocessor(importer.raw_data)
        elif dataset_name == "tatsu-lab/alpaca":
            preprocessor = AlpacaRawDataPreprocessor(importer.raw_data)
        elif dataset_name == "lionelchg/dolly_open_qa":
            preprocessor = DollyOpenQARawDataPreprocessor(importer.raw_data)
        elif dataset_name == "Arsive/toxicity_classification_jigsaw":
            preprocessor = ToxicityDataPreprocessor(importer.raw_data)
        elif dataset_name == "s-nlp/en_paradetox_toxicity":
            preprocessor = ParadetoxDataPreprocessor(importer.raw_data)
        elif dataset_name in ("eriktks/conll2003", "Babelscape/wikineural"):
            preprocessor = NERRawDataPreprocessor(importer.raw_data)
        else:
            preprocessor = RawDataPreprocessor(importer.raw_data)
            warning(f"Using default preprocessor for {dataset_name}")
        try:
            dataset_analysis = preprocessor.analyze()
        except Exception as e:
            print(f"{dataset_name} analysis has some problems!")
            raise e

        analytics = DatasetReferenceDTO(
            dataset_number_of_samples=dataset_analysis["dataset_number_of_samples"],
            dataset_columns=dataset_analysis["dataset_columns"],
            dataset_duplicates=dataset_analysis["dataset_duplicates"],
            dataset_empty_rows=dataset_analysis["dataset_empty_rows"],
            dataset_sample_min_len=dataset_analysis["dataset_sample_min_len"],
            dataset_sample_max_len=dataset_analysis["dataset_sample_max_len"],
        )
        dataset_references.add(dataset_name, analytics)

    dataset_references.dump(destination_path)


if __name__ == "__main__":
    main()
