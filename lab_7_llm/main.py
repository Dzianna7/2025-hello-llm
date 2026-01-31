"""
Laboratory work.

Working with Large Language Models.
"""
from pathlib import Path
# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called
from typing import Iterable, Sequence

import pandas as pd
import torch
from torchinfo import summary
from datasets import load_dataset

# from sympy.codegen import Print
from torch.utils.data import Dataset
from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor
from core_utils.llm.task_evaluator import AbstractTaskEvaluator
from core_utils.llm.time_decorator import report_time

from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

class RawDataImporter(AbstractRawDataImporter):
    """
    A class that imports the HuggingFace dataset.
    """
    @report_time
    def obtain(self) -> None:
        """
        Download a dataset.

        Raises:
            TypeError: In case of downloaded dataset is not pd.DataFrame
        """

        dataset = load_dataset("dair-ai/emotion", split="validation")
        dataframe = dataset.to_pandas()
        self._raw_data = dataframe

        if not isinstance(self._raw_data, pd.DataFrame):
            raise TypeError("Downloaded dataset is not pd.DataFrame")


class RawDataPreprocessor(AbstractRawDataPreprocessor):
    """
    A class that analyzes and preprocesses a dataset.
    """
    def analyze(self) -> dict:
        """
        Analyze a dataset.

        Returns:
            dict: Dataset key properties
        """
        # 1. Number of samples
        num_samples = len(self._raw_data)

        # 2. Number of columns
        num_columns = len(self._raw_data.columns)

        # 3. Number of duplicates
        num_duplicates = int(self._raw_data.duplicated().sum())

        # 4. Number of empty rows
        empty_rows = int(self._raw_data.isna().all(axis=1).sum())

        # 5. Minimum and maximum length
        if self._raw_data.empty:
            min_len = 0
            max_len = 0
        else:
            min_len = float('inf')
            max_len = 0

            for col in self._raw_data.columns:
                lengths = self._raw_data[col].astype(str).str.len()
                if not lengths.empty:
                    min_len = min(min_len, lengths.min())
                    max_len = max(max_len, lengths.max())

            min_len = int(min_len) if min_len != float('inf') else 0

        return {
            "dataset_number_of_samples": num_samples,
            "dataset_columns": num_columns,
            "dataset_duplicates": num_duplicates,
            "dataset_empty_rows": empty_rows,
            "dataset_sample_min_len": min_len,
            "dataset_sample_max_len": max_len
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        # Rename columns
        self._raw_data.rename(columns={'label': 'target', 'text': 'source'}, inplace=True)
        # Reset index
        self._raw_data.reset_index(drop=True, inplace=True)
        self._data = self._raw_data

class TaskDataset(Dataset):
    """
    A class that converts pd.DataFrame to Dataset and works with it.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initialize an instance of TaskDataset.

        Args:
            data (pandas.DataFrame): Original data
        """
        self._data = data

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Returns:
            int: The number of items in the dataset
        """
        return len(self._data)


    def __getitem__(self, index: int) -> tuple[str, ...]:
        """
        Retrieve an item from the dataset by index.

        Args:
            index (int): Index of sample in dataset

        Returns:
            tuple[str, ...]: The item to be received
        """
        row = self._data.iloc[index]
        return tuple(str(value) for value in row)

    @property
    def data(self) -> pd.DataFrame:
        """
        Property with access to preprocessed DataFrame.

        Returns:
            pandas.DataFrame: Preprocessed DataFrame
        """
        return self._data

class LLMPipeline(AbstractLLMPipeline):
    """
    A class that initializes a model, analyzes its properties and infers it.
    """

    def __init__(
        self, model_name: str, dataset: TaskDataset, max_length: int, batch_size: int, device: str
    ) -> None:
        """
        Initialize an instance.

        Args:
            model_name (str): The name of the pre-trained model
            dataset (TaskDataset): The dataset used
            max_length (int): The maximum length of generated sequence
            batch_size (int): The size of the batch inside DataLoader
            device (str): The device for inference
        """
        self._model_name = model_name
        self._dataset = dataset
        self._max_length = max_length
        self._batch_size = batch_size
        self._device = device

        self._tokenizer = AutoTokenizer.from_pretrained("aiknowyou/it-emotion-analyzer")
        self._model = AutoModelForSequenceClassification.from_pretrained("aiknowyou/it-emotion-analyzer").to(device)


    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        input_shape = [1, self._model.config.max_position_embeddings]
        input_data = {
            "input_ids": torch.ones(*input_shape, dtype=torch.long)
        }

        result = summary(self._model, input_data=input_data)

        last_layer = result.summary_list[-1]

        properties = {
            "input_shape": input_shape,
            "embedding_size": getattr(self._model.config, "hidden_size", None),
            "output_shape": list(last_layer.output_size) if hasattr(last_layer.output_size, '__iter__') else [
                last_layer.output_size],
            "num_trainable_params": result.trainable_params,
            "vocab_size": getattr(self._model.config, "vocab_size", None),
            "size": result.total_params,
            "max_context_length": self._model.config.max_position_embeddings
        }

        return properties

    @report_time
    def infer_sample(self, sample: tuple[str, ...]) -> str | None:
        """
        Infer model on a single sample.

        Args:
            sample (tuple[str, ...]): The given sample for inference with model

        Returns:
            str | None: A prediction
        """
        inputs = self._tokenizer(sample[0], return_tensors="pt")

        self._model.eval()

        with torch.no_grad():
            outputs = self._model(**inputs)

        predictions = torch.argmax(outputs.logits).item()

        return str(predictions)

    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """


class TaskEvaluator(AbstractTaskEvaluator):
    """
    A class that compares prediction quality using the specified metric.
    """

    def __init__(self, data_path: Path, metrics: Iterable[Metrics]) -> None:
        """
        Initialize an instance of Evaluator.

        Args:
            data_path (pathlib.Path): Path to predictions
            metrics (Iterable[Metrics]): List of metrics to check
        """

    def run(self) -> dict:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict: A dictionary containing information about the calculated metric
        """
