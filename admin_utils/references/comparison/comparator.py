"""
PythonTool for checking quality of models' and datasets' work
"""

# pylint: disable=import-error

from pathlib import Path

import pandas as pd
from tap import Tap

from admin_utils.constants import DIST_PATH
from admin_utils.references.models import JSONLoader, JSONSchema, MSGStorage, OutputSchema


class CLIArgs(Tap):
    """
    CLI arguments parser to compare references and save result to file
    """

    new_references: str
    old_references: str
    name: str = "compared_before_after_sft"

    @property
    def output_path(self) -> Path:
        """
        Computed path for file to be saved

        Returns:
            Path: Path to output file
        """
        return DIST_PATH / f"{self.name}.csv"


class ModelComparator:
    """
    Class that operates degradations finding
    """

    def __init__(self, current_data: list[JSONSchema], reference_data: list[JSONSchema]) -> None:
        """
        Initialize an instance of ModelComparator class

        Args:
            current_data (list[JSONSchema]): Data of current file info.
            reference_data (list[JSONSchema]): Data of reference file info.
        """
        self.current_data = current_data
        self.reference_data = reference_data

    def map_references(self) -> dict[tuple[str, str], JSONSchema]:
        """
        Method that creates hash map for faster searching for matching pairs between 2 files

        Returns:
            dict[tuple[str, str], JSONSchema]: Reference file hash map.
        """
        return {(r.model, r.dataset): r for r in self.reference_data}

    def find_degradations(self) -> list[OutputSchema]:
        """
        Key method that finds degradations and formats output

        Returns:
            list[OutputSchema]: Massive of degradations search info.
        """
        degradations = []

        reference_map = self.map_references()

        for current_item in self.current_data:
            key = (current_item.model, current_item.dataset)
            reference = reference_map.get(key)

            if not reference:
                degradations.append(
                    OutputSchema(
                        message=MSGStorage.MSG_NOT_COVERED.value,
                        model=current_item.model,
                        dataset=current_item.dataset,
                    )
                )
            elif (degraded_metrics := self._is_worse(current_item.score, reference.score))[0]:
                degradations.append(
                    OutputSchema(
                        model=current_item.model,
                        dataset=current_item.dataset,
                        degraded_metrics=degraded_metrics[1],
                        current_values=degraded_metrics[2],
                        reference_values=degraded_metrics[3],
                    )
                )

        return degradations

    @staticmethod
    def _is_worse(
        current_scores: dict, reference_scores: dict
    ) -> tuple[bool, list[str], dict[str, float], dict[str, float]]:
        """
        Method that checks if degradation is found and collects degraded metrics

        Args:
            current_scores (dict): Scores of current file pairs.
            reference_scores (dict): Scores of reference file pairs.

        Returns:
            tuple[bool, list[str], dict[str, float], dict[str, float]]: Info about degradations.
        """
        degradations = []
        current_values = {}
        reference_values = {}

        for metric, ref_value in reference_scores.items():
            if metric in current_scores:
                if (current_value := current_scores[metric]) < ref_value:
                    degradations.append(metric)
                    current_values[metric] = current_value
                    reference_values[metric] = ref_value

        return bool(degradations), degradations, current_values, reference_values


def compare(current_data_path: Path, reference_data_path: Path) -> list[OutputSchema]:
    """
    Method that compares files to find degradations

    Args:
        current_data_path (Path): Path to current file to be loaded.
        reference_data_path (Path): Path to reference file to be loaded.

    Returns:
        list[OutputSchema]: Results of comparisons.
    """
    current_data = JSONLoader.load(current_data_path)
    reference_data = JSONLoader.load(reference_data_path)

    comparator = ModelComparator(current_data, reference_data)

    return comparator.find_degradations()


def save_to_file(dataframe: pd.DataFrame, output_path: Path) -> None:
    """
    Method to save created dataframe to file

    Args:
        dataframe (pd.DataFrame): Computed data of comparison.
        output_path (Path): Path to file to be saved.
    """
    if output_path.exists():
        output_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, header=True, index=False)


def main() -> None:
    """
    Main method that operates comparison
    """
    args = CLIArgs(underscores_to_dashes=True).parse_args()

    current_data_path = Path(args.new_references)
    compared_data_path = Path(args.old_references)

    degradations = compare(current_data_path, compared_data_path)

    df = pd.DataFrame(item.model_dump() for item in degradations)
    save_to_file(df, args.output_path)


if __name__ == "__main__":
    main()
