"""
Data loading utilities for nanoGPT Extended.

Supports memory-mapped binary files (.bin format, uint16 tokens) — the same
format produced by the prepare_*.py scripts.  Using numpy memmap means the OS
handles paging, so datasets larger than RAM work transparently.
"""

import os
import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------

def get_dataset_path(dataset: str, split: str) -> str:
    """Return the absolute path to a tokenised .bin file.

    Convention: data/<dataset>/<split>.bin

    Args:
        dataset: Dataset name, e.g. ``'shakespeare_char'``, ``'openwebtext'``.
        split:   ``'train'`` or ``'val'``.

    Returns:
        Absolute path string.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, dataset, f"{split}.bin")


# ---------------------------------------------------------------------------
# Batch loader (random-access, suitable for training)
# ---------------------------------------------------------------------------

def get_batch(
    split: str,
    data_dir: str,
    block_size: int,
    batch_size: int,
    device: str,
    device_type: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch from a memory-mapped token file.

    The token file is expected to be a flat array of uint16 values written by
    one of the prepare_*.py scripts.  Each call opens (or re-uses) the memmap
    handle; the OS caches pages, so repeated calls are fast.

    Args:
        split:       ``'train'`` or ``'val'``.
        data_dir:    Directory that contains ``train.bin`` / ``val.bin``.
        block_size:  Sequence length (context window).
        batch_size:  Number of sequences per batch.
        device:      Target device string, e.g. ``'cuda'``, ``'cpu'``.
        device_type: ``'cuda'`` or ``'cpu'`` (used to decide pin_memory).

    Returns:
        ``(x, y)`` where both are ``LongTensor`` of shape ``(batch_size, block_size)``.
        ``y`` is ``x`` shifted by one position — the prediction targets.
    """
    filename = os.path.join(data_dir, f"{split}.bin")
    # Re-open memmap on every call so that file updates are reflected and
    # multiprocessing forks don't share the same file handle.
    data = np.memmap(filename, dtype='uint16', mode='r')
    n = len(data)

    # Sample random starting indices; ensure we can read block_size+1 tokens.
    ix = torch.randint(n - block_size, (batch_size,))

    x = torch.stack([
        torch.from_numpy(data[i: i + block_size].astype(np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i + 1: i + 1 + block_size].astype(np.int64))
        for i in ix
    ])

    if device_type == 'cuda':
        # Pin memory so the host→device transfer can be asynchronous.
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    return x, y


# ---------------------------------------------------------------------------
# Dataset size
# ---------------------------------------------------------------------------

def estimate_dataset_size(data_path: str) -> int:
    """Return the number of tokens stored in a .bin file.

    Args:
        data_path: Path to a uint16 .bin file.

    Returns:
        Integer token count.
    """
    data = np.memmap(data_path, dtype='uint16', mode='r')
    return len(data)


# ---------------------------------------------------------------------------
# PyTorch Dataset wrapper
# ---------------------------------------------------------------------------

class TokenDataset(Dataset):
    """A map-style PyTorch Dataset backed by a memory-mapped uint16 token file.

    Each item is a ``(x, y)`` pair of ``LongTensor`` with shape
    ``(block_size,)``.  Suitable for use with ``torch.utils.data.DataLoader``.

    Args:
        data_path:  Path to a uint16 .bin file.
        block_size: Number of tokens in each input sequence.

    Example::

        ds = TokenDataset('data/shakespeare_char/train.bin', block_size=256)
        loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)
    """

    def __init__(self, data_path: str, block_size: int) -> None:
        super().__init__()
        self.block_size = block_size
        self.data = np.memmap(data_path, dtype='uint16', mode='r')
        # Number of complete (x, y) pairs we can extract.
        self._len = len(self.data) - block_size

        if self._len <= 0:
            raise ValueError(
                f"Dataset at {data_path!r} has {len(self.data)} tokens, "
                f"which is not enough for block_size={block_size}."
            )

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(
            self.data[idx: idx + self.block_size].astype(np.int64)
        )
        y = torch.from_numpy(
            self.data[idx + 1: idx + 1 + self.block_size].astype(np.int64)
        )
        return x, y
