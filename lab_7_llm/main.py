"""
Laboratory work.

Working with Large Language Models.
"""

# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called

# from sympy.codegen import Print
from pathlib import Path
from typing import Iterable, Sequence

import evaluate
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from torchinfo import summary
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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

        dataframe = load_dataset("dair-ai/emotion", split="validation").to_pandas()
        self._raw_data = dataframe

        if not isinstance(self._raw_data, pd.DataFrame):
            raise TypeError("Dataset is not pd.DataFrame")


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

        # 4. Min and max length of the text
        max_len = int(self._raw_data['text'].str.len().max())
        min_len = int(self._raw_data['text'].str.len().min())

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
        text = str(self._data.iloc[index]["source"])
        return (text,)

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
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name,
            num_labels=6
        ).to(device)


        self._model.eval()


    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        config = self._model.config

        input_ids = torch.ones(
            (1, config.max_position_embeddings),
            dtype=torch.long
        )

        attention_mask = torch.ones_like(input_ids)

        tokens = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        stats = summary(
            self._model,
            input_data=tokens,
            device=self._device,
            verbose=0
        )

        input_shape_dict = {}
        for key, value in stats.input_size.items():
            input_shape_dict[key] = list(value)

        return {
            "input_shape": input_shape_dict,
            "embedding_size": config.max_position_embeddings,
            "output_shape": stats.summary_list[-1].output_size,
            "num_trainable_params": stats.trainable_params,
            "vocab_size": config.vocab_size,
            "size": stats.total_param_bytes,
            "max_context_length": config.max_length
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
        if not sample or len(sample) == 0:
            return None

        text = sample[0]

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=self._max_length,
            add_special_tokens=True
        )

        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

            max_prob = torch.max(probabilities).item()
            if max_prob < 0.5:  # Порог уверенности
                return "3"

            predicted_class = torch.argmax(logits, dim=-1).item()

        return str(predicted_class)

    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """
        predictions = []
        # Проходим по всему датасету без DataLoader
        for i in range(len(self._dataset)):
            # Получаем сэмпл
            sample = self._dataset[i]
            # Предсказываем для одного сэмпла
            prediction = self.infer_sample(sample)
            predictions.append(prediction)

        # Создаем DataFrame с результатами
        result_df = self._dataset.data.copy()
        result_df['predictions'] = predictions

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
        texts = [sample[0] for sample in sample_batch]

        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self._max_length
        )

        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        self._model.eval()
        outputs = self._model(**inputs)

        predicted_classes = torch.argmax(outputs.logits, dim=-1).tolist()

        return [self._model.config.id2label[c] for c in predicted_classes]


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
        df = pd.read_csv(self._data_path)

        predictions = df["predictions"].astype(int).tolist()
        references = df["target"].astype(int).tolist()

        metric = evaluate.load("f1")

        score = metric.compute(
            predictions=predictions,
            references=references,
            average="weighted"
        )

        return {"f1": score["f1"]}