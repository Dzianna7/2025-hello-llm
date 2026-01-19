"""
Collect and store model analytics.
"""

# pylint: disable=assignment-from-no-return

from pathlib import Path
from typing import Any

import simplejson as json
from tqdm import tqdm

try:
    from pandas import DataFrame
except ImportError:
    print('Library "pandas" not installed. Failed to import.')
    DataFrame = dict  # type: ignore

from admin_utils.constants import DEVICE
from lab_7_llm.main import LLMPipeline, TaskDataset


def get_references(path: Path) -> Any:
    """
    Load reference_scores.json file.

    Args:
        path (Path): Path to file

    Returns:
        Any: File content
    """
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def save_reference(path: Path, refs: dict) -> None:
    """
    Save analytics.

    Args:
        path (Path): Path to file with analytics
        refs (dict): Model analysis for models
    """
    with open(path, mode="w", encoding="utf-8") as file:
        json.dump(refs, file, indent=4, ensure_ascii=False, sort_keys=True, use_decimal=True)
    with open(path, mode="a", encoding="utf-8") as file:
        file.write("\n")


def main() -> None:
    """
    Run collected models analytics.
    """
    batch_size = 64
    max_length = 120
    device = DEVICE

    references_dir = Path(__file__).parent / "gold"
    references_path = references_dir / "reference_scores.json"
    destination_path = references_dir / "reference_model_analytics.json"

    references = get_references(path=references_path)
    result = {}

    for model_name in tqdm(sorted(references)):
        print(model_name, flush=True)
        pipeline = LLMPipeline(
            model_name, TaskDataset(DataFrame([])), max_length, batch_size, device
        )
        model_analysis = pipeline.analyze_model()
        result[model_name] = model_analysis
    save_reference(destination_path, result)


if __name__ == "__main__":
    main()
