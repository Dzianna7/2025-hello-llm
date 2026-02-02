from pathlib import Path
from typing import Any, Callable, Protocol

import torch  # type: ignore
from torch.utils.data.dataset import Dataset

class EvalPrediction:
    logits: torch.Tensor  # type: ignore
    labels: list[str]

    def __getitem__(self, index: int) -> torch.Tensor | list[str]:  # type: ignore
        return (self.logits, self.labels)[index]

class TrainingArguments:
    def __init__(
        self,
        output_dir: Path | str,
        learning_rate: float,
        per_device_train_batch_size: int,
        max_steps: int,
        use_cpu: bool,
        save_strategy: str,
        load_best_model_at_end: bool,
        remove_unused_columns: bool = True,
        seed: int = 42,
    ): ...

class Trainer:
    def __init__(  # type: ignore
        self,
        model: torch.nn.Module,
        args: TrainingArguments,
        train_dataset: Dataset | list[dict[str, Any]],
        data_collator: Callable[[AutoTokenizer], torch.Tensor] | None = None,
    ): ...
    def save_model(self, path: Path) -> None: ...
    def train(self) -> None: ...

class BatchEncoding:
    input_features: torch.Tensor  # type: ignore
    def to(self, device: str) -> BatchEncoding: ...
    def __getitem__(self, el: str) -> Any: ...
    def __setitem__(self, el: str, val: Any) -> None: ...
    def keys(self) -> list: ...

class LayerLike(Protocol):
    out_features: int

class ContainerLike(Protocol):
    dense: LayerLike
    out_proj: torch.nn.Linear  # type: ignore

    def parameters(self) -> list: ...

class AutoModel:
    classifier: ContainerLike
    pooler: ContainerLike
    longformer: ContainerLike
    num_labels: int

    @staticmethod
    def from_pretrained(model_name: str) -> AutoModel: ...
    def load_state_dict(self, state_dict: dict, strict: bool, assign: bool) -> None: ...
    def state_dict(self) -> dict: ...
    def __call__(self, *args: Any, **kwds: Any) -> torch.Tensor: ...  # type: ignore
    def to(self, device: str) -> AutoModel: ...
    def eval(self) -> AutoModel: ...
    def generate(self, values: torch.Tensor) -> list: ...  # type: ignore

class PreTrainedTokenizerBase:
    cls_token_id: int
    mask_token_id: int

class AutoTokenizer(PreTrainedTokenizerBase):
    def __call__(self, *args: Any, **kwds: Any) -> BatchEncoding: ...
