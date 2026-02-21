"""
Fine-tuning starter.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
import json
from pathlib import Path

from core_utils.llm.metrics import Metrics
from core_utils.project.lab_settings import LabSettings
from lab_8_sft.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    report_time,
    TaskDataset,
    TaskEvaluator,
)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    base_dir = Path(__file__).resolve().parent
    config = LabSettings(base_dir / "settings.json")

    data_loader = RawDataImporter(config.parameters.dataset)
    data_loader.obtain()

    if data_loader.raw_data is None:
        return

    processor = RawDataPreprocessor(data_loader.raw_data)
    processor.transform()

    prepared_df = processor.data.head(100)
    task_data = TaskDataset(prepared_df)

    llm = LLMPipeline(
        model_name=config.parameters.model,
        dataset=task_data,
        max_length=120,
        batch_size=64,
        device="cpu",
    )

    analysis = llm.analyze_model()

    print("\nModel analysis:")
    for name in analysis:
        print(f"{name}: {analysis[name]}")

    example = task_data[0]
    generated = llm.infer_sample(example)

    print("\nSample inference:")
    print(example[0])
    print("Translation:", example[1])
    print("Prediction:", generated)

    output_dir = base_dir / "dist"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "predictions.csv"

    results_df = llm.infer_dataset()
    results_df.to_csv(output_file)

    metric_objects = [Metrics(m) for m in config.parameters.metrics]
    task_eval = TaskEvaluator(output_file, metric_objects)

    print("\nEvaluation:")
    evaluation_scores = task_eval.run()
    for metric_name, score in evaluation_scores.items():
        print(f"{metric_name}: {score}")

    result = task_eval
    assert result is not None, "Demo does not work correctly"

if __name__ == "__main__":
    main()
