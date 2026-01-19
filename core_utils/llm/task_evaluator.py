"""
Module with description of abstract task evaluator.
"""

# pylint: disable=too-few-public-methods, duplicate-code
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from core_utils.llm.metrics import Metrics


class AbstractTaskEvaluator(ABC):
    """
    Abstract Task Evaluator.
    """

    def __init__(self, data_path: Path, metrics: Iterable[Metrics]) -> None:
        """
        Initialize an instance of AbstractTaskEvaluator.

        Args:
            data_path (pathlib.Path): Path to predictions
            metrics (Iterable[Metrics]): List of metrics to check
        """
        self._data_path = data_path
        self._metrics = metrics

    @abstractmethod
    def run(self) -> dict[str, float]:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict[str, float]: A dictionary containing information about the calculated metric
        """
