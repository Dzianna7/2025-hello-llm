"""
Torch data load listing.
"""

# pylint: disable=duplicate-code

try:
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    print('Library "torch" not installed. Failed to import.')
    DataLoader = None  # type: ignore
    Dataset = None  # type: ignore

try:
    from pandas import DataFrame
except ImportError:
    print('Library "pandas" not installed. Failed to import.')
    DataFrame = None  # type: ignore

try:
    from datasets import load_dataset
except ImportError:
    print('Library "datasets" not installed. Failed to import.')
    load_dataset = None  # type: ignore


class TaskDataset(Dataset):  # type: ignore
    """
    Dataset with translation data.
    """

    def __init__(self, data: DataFrame) -> None:
        """
        Initialize an instance of TaskDataset.

        Args:
            data (pandas.DataFrame): original data.
        """
        self._data = data

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
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
        return (str(self._data["neutral"].iloc[index]),)


def main() -> None:
    """
    Entrypoint for the listing.
    """

    # 1. Load dataset
    data = load_dataset("s-nlp/ru_paradetox_toxicity", split="train").to_pandas()
    dataset = TaskDataset(data.loc[:100])

    # 2. Get data loader with batch 1
    dataset_loader = DataLoader(dataset)
    print(len(dataset_loader))

    # 3. Print result
    print(next(iter(dataset_loader)))
    print(len(next(iter(dataset_loader))[0]))

    # 4. Get data loader with batch 4
    dataset_loader = DataLoader(dataset, batch_size=4)
    print(len(dataset_loader))

    # 5.
    print(next(iter(dataset_loader)))
    print(len(next(iter(dataset_loader))[0]))


if __name__ == "__main__":
    main()
