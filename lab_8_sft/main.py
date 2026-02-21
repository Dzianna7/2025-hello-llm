"""
Laboratory work.

Working with Large Language Models.
"""

from pathlib import Path

# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called
from typing import Iterable, Sequence

import evaluate
import pandas as pd
import torch
from datasets import load_dataset
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor, ColumnNames
from core_utils.llm.task_evaluator import AbstractTaskEvaluator
from core_utils.llm.time_decorator import report_time


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
        self._raw_data = load_dataset(self._hf_name, split='test').to_pandas()

        if not isinstance(self._raw_data, pd.DataFrame):
            raise TypeError("The downloaded dataset is not pd.DataFrame")


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
        raw_data_copy = self._raw_data["EN"].dropna()

        return {
            'dataset_number_of_samples': len(self._raw_data),
            'dataset_columns': len(self._raw_data.columns),
            'dataset_duplicates': self._raw_data.duplicated().sum(),
            'dataset_empty_rows': self._raw_data.isna().sum().sum(),
            'dataset_sample_min_len': raw_data_copy.str.len().min(),
            'dataset_sample_max_len': raw_data_copy.str.len().max(),
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data
        self._data = self._data.rename(columns={'EN': ColumnNames.SOURCE, 'DE': ColumnNames.TARGET})
        self._data = self._data.drop_duplicates(subset=[ColumnNames.SOURCE])
        self._data[ColumnNames.SOURCE] = ("Translate from English to German: " +
                                          self._data[ColumnNames.SOURCE].astype(str))
        self._data = self._data.reset_index(drop=True)


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
        return (str(self._data[ColumnNames.SOURCE].iloc[index]),
                str(self._data[ColumnNames.TARGET].iloc[index]))

    @property
    def data(self) -> DataFrame:
        """
        Property with access to preprocessed DataFrame.

        Returns:
            pandas.DataFrame: Preprocessed DataFrame
        """
        return self._data
#
#
# def tokenize_sample(
#     sample: pd.Series, tokenizer: AutoTokenizer, max_length: int
# ) -> dict[str, torch.Tensor]:
#     """
#     Tokenize sample.
#
#     Args:
#         sample (pandas.Series): sample from a dataset
#         tokenizer (transformers.models.auto.tokenization_auto.AutoTokenizer): Tokenizer to tokenize
#             original data
#         max_length (int): max length of sequence
#
#     Returns:
#         dict[str, torch.Tensor]: Tokenized sample
#     """
#
#
# class TokenizedTaskDataset(Dataset):
#     """
#     A class that converts pd.DataFrame to Dataset and works with it.
#     """
#
#     def __init__(self, data: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int) -> None:
#         """
#         Initialize an instance of TaskDataset.
#
#         Args:
#             data (pandas.DataFrame): Original data
#             tokenizer (transformers.models.auto.tokenization_auto.AutoTokenizer): Tokenizer to
#                 tokenize the dataset
#             max_length (int): max length of a sequence
#         """
#
#     def __len__(self) -> int:
#         """
#         Return the number of items in the dataset.
#
#         Returns:
#             int: The number of items in the dataset
#         """
#
#     def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
#         """
#         Retrieve an item from the dataset by index.
#
#         Args:
#             index (int): Index of sample in dataset
#
#         Returns:
#             dict[str, torch.Tensor]: An element from the dataset
#         """
#

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
        self._model = model_name
        self._dataset = dataset
        self._max_length = max_length
        self._device = device
        self._batch_size = batch_size

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._model.to(self._device)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        model = self._model
        if not isinstance(model, torch.nn.Module):
            return {}

        config = model.config

        dummy_input = torch.zeros(
            (self._batch_size, self._max_length),
            dtype=torch.long,
            device=self._device
        )

        stats = summary(
            model,
            input_data={
                "input_ids": dummy_input,
                "decoder_input_ids": dummy_input
            },
            verbose=0
        )

        return {
            "input_shape": [1, config.d_model],
            "embedding_size": config.d_model,
            "output_shape": [1, config.d_model, config.vocab_size],
            "num_trainable_params": int(stats.trainable_params),
            "vocab_size": config.vocab_size,
            "size": int(stats.total_param_bytes),
            "max_context_length": config.max_length,
        }

    @report_time
    def infer_sample(self, sample: tuple[str, ...]) -> str | None:
        """
        Infer model on a single sample.

        Args:
            sample (tuple[str, ...]): The given sample for inference with model

        Returns:
            str | None: A prediction
        """
        if self._model is None:
            return None

        result = self._infer_batch([sample])
        return result[0] if result else None

    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """
        predictions: list[str] = []
        references: list[str] = []

        loader = DataLoader(self._dataset, batch_size=self._batch_size)

        for batch in loader:
            sources, targets = batch
            batch_predictions = self._infer_batch(sources)

            predictions += batch_predictions
            references += list(targets)

        result_df = pd.DataFrame({
            ColumnNames.TARGET.value: references,
            ColumnNames.PREDICTION.value: predictions
        })

        return result_df

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """
        model = self._model
        if model is None:
            return []

        if isinstance(sample_batch[0], str):
            texts = list(sample_batch)
        else:
            texts = [item[0] for item in sample_batch]

        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )

        encoded = {name: tensor.to(self._device) for name, tensor in encoded.items()}

        model.eval()
        generated_ids = model.generate(
            **encoded,
            max_length=self._max_length
        )

        decoded_texts = self._tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        return [str(text) for text in decoded_texts]


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
        self._data_path = data_path
        self._metrics = metrics

    def run(self) -> dict:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict: A dictionary containing information about the calculated metric
        """
        data = pd.read_csv(self._data_path)

        return {
            metric.value: evaluate.load(metric.value).compute(
                predictions=data[ColumnNames.PREDICTION.value],
                references=data[ColumnNames.TARGET.value])["bleu"]
            for metric in self._metrics
        }


# class SFTPipeline(AbstractSFTPipeline):
#     """
#     A class that initializes a model, fine-tuning.
#     """
#
#     def __init__(
#         self,
#         model_name: str,
#         dataset: Dataset,
#         sft_params: SFTParams,
#         data_collator: Callable[[AutoTokenizer], torch.Tensor] | None = None,
#     ) -> None:
#         """
#         Initialize an instance of ClassificationSFTPipeline.
#
#         Args:
#             model_name (str): The name of the pre-trained model.
#             dataset (torch.utils.data.dataset.Dataset): The dataset used.
#             sft_params (SFTParams): Fine-Tuning parameters.
#             data_collator (Callable[[AutoTokenizer], torch.Tensor] | None, optional): processing
#                                                                     batch. Defaults to None.
#         """
#
#     def run(self) -> None:
#         """
#         Fine-tune model.
#         """
