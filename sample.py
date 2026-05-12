"""
Text generation / inference script for nanoGPT Extended.

Loads a trained model from a checkpoint directory or directly from a GPT-2
pretrained weight and samples continuations of a given prompt.

Usage:
    python sample.py --out_dir=out --start="Hello, world"
    python sample.py --init_from=gpt2 --start="Once upon a time"
    python sample.py --init_from=resume --start="FILE:prompt.txt"
"""

import os
import pickle
import contextlib
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Top-level config variables (can be overridden by configurator.py)
# ---------------------------------------------------------------------------

init_from: str = 'resume'       # 'resume' | 'gpt2' | 'gpt2-medium' | 'gpt2-large' | 'gpt2-xl'
out_dir: str = 'out'            # directory with ckpt.pt (only used when init_from='resume')
start: str = "\n"               # prompt string; prefix with 'FILE:' to read from a file
num_samples: int = 10           # number of samples to generate
max_new_tokens: int = 500       # tokens generated per sample
temperature: float = 0.8        # 1.0 = no scaling, < 1 = sharper, > 1 = flatter
top_k: int = 200                # keep only top-k logits (0 = disabled)
top_p: Optional[float] = None   # nucleus sampling probability threshold (None = disabled)
seed: int = 1337
device: str = 'auto'            # 'auto' | 'cpu' | 'cuda' | 'cuda:0' | 'mps' etc.
dtype: str = 'bfloat16'         # 'float32' | 'bfloat16' | 'float16'
compile: bool = False           # torch.compile the model for faster sampling

# ---------------------------------------------------------------------------
# Allow overriding config from command line via configurator.py
# ---------------------------------------------------------------------------
import sys

exec(open(os.path.join(os.path.dirname(__file__), 'configurator.py')).read()) if os.path.exists(
    os.path.join(os.path.dirname(__file__), 'configurator.py')
) else None

# Apply any --key=value arguments directly
for arg in sys.argv[1:]:
    if arg.startswith('--'):
        key, _, val = arg[2:].partition('=')
        if key in dir():
            try:
                exec(f"{key} = {val}")
            except Exception:
                exec(f"{key} = '{val}'")

# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------

if device == 'auto':
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

device_type = 'cuda' if device.startswith('cuda') else device

torch.manual_seed(seed)
if device_type == 'cuda':
    torch.cuda.manual_seed(seed)

# Precision context
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = (
    contextlib.nullcontext()
    if device_type in ('cpu', 'mps')
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

from model import GPT, GPTConfig  # noqa: E402

if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # Strip any DDP '_orig_mod.' prefix added by torch.compile / DDP
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
else:
    raise ValueError(f"Unknown init_from: {init_from!r}. Use 'resume' or a gpt2 variant.")

model.eval()
model.to(device)

if compile:
    model = torch.compile(model)

# ---------------------------------------------------------------------------
# Tokenizer setup
# ---------------------------------------------------------------------------
# Prefer tiktoken (GPT-2 BPE) when loading from a pretrained GPT-2.
# Fall back to a character-level tokenizer when a meta.pkl is present in
# out_dir (created by data/prepare_shakespeare.py).

encode = None
decode = None

if init_from == 'resume':
    meta_path = os.path.join(out_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        print(f"Loading character-level tokenizer from {meta_path}")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi: dict = meta['stoi']
        itos: dict = meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        # Assume GPT-2 BPE tokenizer
        try:
            import tiktoken
            enc = tiktoken.get_encoding('gpt2')
            encode = lambda s: enc.encode(s, allowed_special={'<|endoftext|>'})
            decode = lambda l: enc.decode(l)
        except ImportError:
            print("tiktoken not found; using character-level fallback (single characters).")
            encode = lambda s: [ord(c) for c in s]
            decode = lambda l: ''.join([chr(i) for i in l])
else:
    # GPT-2 pretrained
    try:
        import tiktoken
        enc = tiktoken.get_encoding('gpt2')
        encode = lambda s: enc.encode(s, allowed_special={'<|endoftext|>'})
        decode = lambda l: enc.decode(l)
    except ImportError:
        print("tiktoken not found; using character-level fallback (single characters).")
        encode = lambda s: [ord(c) for c in s]
        decode = lambda l: ''.join([chr(i) for i in l])

# ---------------------------------------------------------------------------
# Build the starting context tensor
# ---------------------------------------------------------------------------

if start.startswith('FILE:'):
    filepath = start[len('FILE:'):]
    with open(filepath, 'r', encoding='utf-8') as f:
        start = f.read()

start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------

print(f"\n--- Generating {num_samples} samples (max {max_new_tokens} tokens each) ---\n")

with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(
                x,
                max_new_tokens,
                temperature=temperature,
                top_k=top_k if top_k > 0 else None,
                top_p=top_p,
            )
            print(decode(y[0].tolist()))
            print('=' * 80)
