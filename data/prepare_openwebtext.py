"""
Prepare the OpenWebText dataset using HuggingFace datasets + tiktoken.

Downloads OpenWebText via HuggingFace `datasets`, tokenises it with the GPT-2
BPE encoder from tiktoken, and writes:

    data/openwebtext/train.bin   (uint16 numpy array, ~17 GB)
    data/openwebtext/val.bin     (uint16 numpy array, ~8.5 MB)

A 0.1 % hold-out of the roughly 8 million OpenWebText documents is used as
the validation split (identical to the original nanoGPT setup).

Requirements:
    pip install datasets tiktoken tqdm

Usage:
    python data/prepare_openwebtext.py
"""

import os
import multiprocessing
from functools import partial

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VAL_FRACTION = 0.001        # 0.1 % of documents for validation
NUM_PROC = max(1, multiprocessing.cpu_count() // 2)  # worker processes
DTYPE = np.uint16           # GPT-2 vocab size < 2**16

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'openwebtext')


# ---------------------------------------------------------------------------
# Tokenisation worker
# ---------------------------------------------------------------------------

def _tokenise_shard(examples: dict, enc) -> dict:
    """Tokenise a batch of documents and append an <|endoftext|> separator.

    This function is designed to be called via ``dataset.map`` with
    ``batched=True``.
    """
    eot = enc.eot_token
    ids = []
    for text in examples['text']:
        ids.extend(enc.encode_ordinary(text))
        ids.append(eot)
    return {'ids': ids, 'len': [len(ids)]}


def _write_shard(
    split_dataset,
    output_path: str,
    desc: str,
) -> int:
    """Write all token ids in *split_dataset* to a flat uint16 binary file.

    Returns the total token count.
    """
    # Compute total length so we can pre-allocate.
    total_len = int(np.sum(split_dataset['len']))
    arr = np.empty(total_len, dtype=DTYPE)

    idx = 0
    for ids in split_dataset['ids']:
        arr_chunk = np.array(ids, dtype=DTYPE)
        arr[idx: idx + len(arr_chunk)] = arr_chunk
        idx += len(arr_chunk)

    arr.tofile(output_path)
    print(f"{desc}: {total_len:,} tokens → {output_path} "
          f"({arr.nbytes / 1e9:.2f} GB)")
    return total_len


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required. "
            "Install it with: pip install datasets"
        )

    try:
        import tiktoken
    except ImportError:
        raise ImportError(
            "The 'tiktoken' package is required. "
            "Install it with: pip install tiktoken"
        )

    try:
        from tqdm import tqdm  # noqa: F401 — used inside datasets.map
    except ImportError:
        pass  # tqdm is optional; datasets will fall back to plain progress

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load raw dataset from HuggingFace Hub
    # ------------------------------------------------------------------
    print("Loading OpenWebText from HuggingFace Hub …")
    dataset = load_dataset('openwebtext', num_proc=NUM_PROC)

    # ------------------------------------------------------------------
    # 2. Train / val split
    # ------------------------------------------------------------------
    split_dataset = dataset['train'].train_test_split(
        test_size=VAL_FRACTION,
        seed=2357,
        shuffle=True,
    )
    # Rename 'test' → 'val' for consistency.
    split_dataset['val'] = split_dataset.pop('test')

    print(f"Train documents: {len(split_dataset['train']):,}")
    print(f"Val documents  : {len(split_dataset['val']):,}")

    # ------------------------------------------------------------------
    # 3. Tokenise with tiktoken (GPT-2 BPE)
    # ------------------------------------------------------------------
    enc = tiktoken.get_encoding('gpt2')

    tokenise = partial(_tokenise_shard, enc=enc)

    print(f"Tokenising with {NUM_PROC} worker(s) …")
    tokenised = split_dataset.map(
        tokenise,
        batched=True,
        remove_columns=['text'],
        num_proc=NUM_PROC,
        desc='Tokenising',
    )

    # ------------------------------------------------------------------
    # 4. Write .bin files
    # ------------------------------------------------------------------
    train_tokens = _write_shard(
        tokenised['train'],
        os.path.join(OUTPUT_DIR, 'train.bin'),
        desc='Train',
    )
    val_tokens = _write_shard(
        tokenised['val'],
        os.path.join(OUTPUT_DIR, 'val.bin'),
        desc='Val',
    )

    print(f"\nDone. Total tokens: {train_tokens + val_tokens:,}")
    print(f"  Train: {train_tokens:,}  Val: {val_tokens:,}")


if __name__ == '__main__':
    main()
