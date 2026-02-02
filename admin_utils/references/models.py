"""
Models for references comparison tool
"""

from enum import StrEnum
from pathlib import Path

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
