"""
Prepare the Shakespeare dataset for character-level language modeling.

Downloads the Tiny Shakespeare dataset (1.1 MB), builds a character-level
vocabulary, encodes the text to integers, splits 90 / 10 into train and
validation sets, and writes:

    data/shakespeare_char/train.bin   (uint16 numpy array)
    data/shakespeare_char/val.bin     (uint16 numpy array)
    data/shakespeare_char/meta.pkl    (vocab_size, itos, stoi)

Usage:
    python data/prepare_shakespeare.py
"""

import os
import pickle
import urllib.request

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_URL = (
    'https://raw.githubusercontent.com/karpathy/char-rnn/master/'
    'data/tinyshakespeare/input.txt'
)
TRAIN_FRACTION = 0.9

# Output directory is always relative to this script's location so the script
# works regardless of the working directory from which it is invoked.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'shakespeare_char')


def download_text(url: str, dest_path: str) -> str:
    """Download *url* to *dest_path* if not already present.

    Returns:
        The full text content of the file.
    """
    if not os.path.exists(dest_path):
        print(f"Downloading {url} …")
        urllib.request.urlretrieve(url, dest_path)
        print(f"Saved to {dest_path}")
    else:
        print(f"Found existing file at {dest_path}; skipping download.")

    with open(dest_path, 'r', encoding='utf-8') as f:
        return f.read()


def build_vocab(text: str) -> tuple[dict, dict]:
    """Build character-level vocabulary from *text*.

    Returns:
        ``(stoi, itos)`` where ``stoi`` maps characters to integers and
        ``itos`` maps integers back to characters.
    """
    chars = sorted(set(text))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size} unique characters")
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


def encode(text: str, stoi: dict) -> list[int]:
    """Encode *text* to a list of integer token ids."""
    return [stoi[c] for c in text]


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Download
    # ------------------------------------------------------------------
    raw_path = os.path.join(OUTPUT_DIR, 'input.txt')
    text = download_text(DATA_URL, raw_path)
    print(f"Total characters in dataset: {len(text):,}")

    # ------------------------------------------------------------------
    # 2. Build vocabulary
    # ------------------------------------------------------------------
    stoi, itos = build_vocab(text)
    vocab_size = len(stoi)

    # ------------------------------------------------------------------
    # 3. Encode
    # ------------------------------------------------------------------
    data = encode(text, stoi)
    print(f"Total tokens (characters): {len(data):,}")

    # ------------------------------------------------------------------
    # 4. Train / val split
    # ------------------------------------------------------------------
    split_idx = int(len(data) * TRAIN_FRACTION)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    print(f"Train tokens : {len(train_data):,}")
    print(f"Val tokens   : {len(val_data):,}")

    # ------------------------------------------------------------------
    # 5. Save binary files
    # ------------------------------------------------------------------
    train_arr = np.array(train_data, dtype=np.uint16)
    val_arr = np.array(val_data, dtype=np.uint16)

    train_path = os.path.join(OUTPUT_DIR, 'train.bin')
    val_path = os.path.join(OUTPUT_DIR, 'val.bin')

    train_arr.tofile(train_path)
    val_arr.tofile(val_path)
    print(f"Saved train.bin ({train_arr.nbytes / 1e6:.2f} MB)")
    print(f"Saved val.bin   ({val_arr.nbytes / 1e6:.2f} MB)")

    # ------------------------------------------------------------------
    # 6. Save metadata
    # ------------------------------------------------------------------
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    meta_path = os.path.join(OUTPUT_DIR, 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    print(f"Saved meta.pkl (vocab_size={vocab_size})")
    print("Done.")


if __name__ == '__main__':
    main()
