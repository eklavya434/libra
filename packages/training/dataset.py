"""
Libra Training - Dataset & Batch Loader
Prepares token tensors and samples random mini-batches for autoregressive next-token prediction.
"""

import torch

from packages.models.generation import encode_string


class TextDataset:
    """In-memory dataset that converts text into token tensors and yields (x, y) mini-batches."""

    def __init__(self, text: str, train_ratio: float = 0.9) -> None:
        raw_tokens = encode_string(text)
        if len(raw_tokens) < 10:
            raise ValueError("Corpus is too short for training a language model.")

        self.data_tensor = torch.tensor(raw_tokens, dtype=torch.long)
        n_train = int(len(self.data_tensor) * train_ratio)
        self.train_data = self.data_tensor[:n_train]
        self.val_data = self.data_tensor[n_train:]

    def get_batch(
        self,
        split: str = "train",
        batch_size: int = 4,
        block_size: int = 64,
        device: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Samples a random mini-batch of inputs (x) and targets (y) shifted by 1."""
        data = self.train_data if split == "train" else self.val_data
        if len(data) <= block_size:
            # Fallback if split is very small
            data = self.data_tensor

        # Random start indices for each batch item
        max_idx = max(1, len(data) - block_size - 1)
        ix = torch.randint(0, max_idx, (batch_size,))

        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

        return x.to(device), y.to(device)
