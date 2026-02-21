"""
Models for references comparison tool
"""

from enum import StrEnum
from pathlib import Path

import simplejson as json
from pydantic import BaseModel, ConfigDict, Field, field_validator, RootModel


class MSGStorage(StrEnum):
    """
    Storage for messages
    """

    MSG_DEGRADATION = "DEGRADED"
    MSG_NOT_COVERED = "CURRENT REFERENCE NOT COVERED"
    MSG_NO_DEGRADATION = "NO DEGRADATION"


class OutputSchema(BaseModel):
    """
    Schema that stores output information to be loaded
    """

    message: str = Field(default=MSGStorage.MSG_DEGRADATION.value)
    model: str
    dataset: str
    degraded_metrics: list[str] = Field(default=[MSGStorage.MSG_NO_DEGRADATION.value])
    current_values: dict[str, float] = Field(default_factory=dict)
    reference_values: dict[str, float] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid", str_min_length=1)


class JSONSchema(BaseModel):
    """
    Schema that contains info about model, its dataset and score
    """

    model: str
    dataset: str
    score: dict[str, float]
    model_config = ConfigDict(extra="forbid", str_min_length=1)

    @classmethod
    @field_validator("score")
    def validate_score(cls, v: dict) -> dict:
        """
        Validator of score field

        Args:
            v (dict): Field of score.

        Returns:
            dict: Field itself.
        """
        if not v:
            raise ValueError("Score must be filled")
        for value in v.values():
            if not isinstance(value, (int, float)):
                raise ValueError("Score must be number")
            if value < 0:
                raise ValueError("Score must be positive number")
        return v


class JSONLoader(RootModel[dict[str, dict[str, dict[str, float]]]]):
    """
    Loader of JSON files via pydantic
    """

    @classmethod
    def from_file(cls, filepath: Path) -> "JSONLoader":
        """
        Method that loads file for further comparison

        Args:
            filepath (Path): Path to file to be loaded.

        Returns:
            JSONLoader: Object for further converting to schema.
        """
        with open(filepath, "r", encoding="utf-8") as file:
            return cls.model_validate_json(file.read())

    def to_schemas(self) -> list[JSONSchema]:
        """
        Method that converts file info into schemas

        Returns:
            list[JSONSchema]: Schemas of model-dataset-score info.
        """
        return [
            JSONSchema(model=model_name, dataset=dataset_name, score=score)
            for model_name, further_info in self.root.items()
            for dataset_name, score in further_info.items()
        ]

    @classmethod
    def load(cls, filepath: Path) -> list[JSONSchema]:
        """
        Method that loads and parses json file in one step

        Args:
            filepath (Path): Path to file to be loaded.

        Returns:
            list[JSONSchema]: Schemas of model-dataset-score info.
        """
        loader = cls.from_file(filepath)
        return loader.to_schemas()


class JSONSerializableMixin:  # pylint: disable=R0903
    """
    Mixin for serializable pydantic models.
    """

    def dump(self, json_path: Path) -> None:
        """
        Save model to JSON.

        Args:
            json_path (Path): Path to the file
        """
        with json_path.open(mode="w", encoding="utf-8") as f:
            content = json.loads(self.model_dump_json())
            json.dump(content, f, indent=4, ensure_ascii=False, sort_keys=True)
            f.write("\n")


class DatasetReferenceDTO(BaseModel):
    """
    Data transfer object for a single dataset's analytics.
    """

    dataset_number_of_samples: int
    dataset_columns: int
    dataset_duplicates: int
    dataset_empty_rows: int
    dataset_sample_min_len: int
    dataset_sample_max_len: int


class DatasetReferencesModel(RootModel[dict[str, DatasetReferenceDTO]], JSONSerializableMixin):
    """
    Model for storing multiple dataset references.
    """

    root: dict[str, DatasetReferenceDTO] = {}

    def add(self, dataset_name: str, analytics: DatasetReferenceDTO) -> None:
        """
        Add dataset to storage

        Args:
            dataset_name (str): Name of dataset
            analytics (DatasetReferenceDTO): Dataset analytics
        """
        self.root[dataset_name] = analytics


class EvaluationReferencesModel(BaseModel):
    """
    Model for loading evaluation references from JSON.
    """

    references: dict

    @classmethod
    def from_json(cls, json_path: Path) -> "EvaluationReferencesModel":
        """
        Load references from JSON file.

        Args:
            json_path (Path): Path to the reference file

        Returns:
            EvaluationReferencesModel: Loaded references
        """
        with json_path.open(encoding="utf-8") as f:
            data = json.load(f)
        return cls(references=data)

    def get_datasets(self) -> list[str]:
        """
        Extract unique dataset names from references.

        Returns:
            list[str]: Sorted list of unique dataset names
        """
        datasets_raw = []
        for _, dataset_pack in self.references.items():
            for dataset_name in dataset_pack.keys():
                datasets_raw.append(dataset_name)
        return sorted(set(datasets_raw))
