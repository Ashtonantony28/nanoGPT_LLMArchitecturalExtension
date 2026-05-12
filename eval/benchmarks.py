"""
Zero-shot benchmark evaluation for nanoGPT Extended.

Provides:
- ``evaluate_lambada``: next-word prediction accuracy on LAMBADA.
- ``evaluate_text_completion``: generate completions for a list of prompts.
- ``run_perplexity_benchmark``: compute token-level perplexity on a text file.

All functions are self-contained and do not require external evaluation
libraries.  They accept any model that implements a ``forward(x, y)``
method returning ``(logits, loss)`` and an optional ``generate`` method.

References:
- LAMBADA: Paperno et al. (2016) https://arxiv.org/abs/1606.06031
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_encode_decode(
    tokenizer,
) -> Tuple[Callable[[str], List[int]], Callable[[List[int]], str]]:
    """Return (encode, decode) callables from a tokenizer object.

    Supports three shapes of tokenizer:
    1. ``tiktoken`` encoder: has ``.encode()`` and ``.decode()`` methods.
    2. Object with callable ``.encode`` / ``.decode`` attributes.
    3. Tuple/pair ``(encode_fn, decode_fn)`` passed directly.

    Args:
        tokenizer: A tokenizer object or a ``(encode, decode)`` tuple.

    Returns:
        ``(encode, decode)`` callables.

    Raises:
        TypeError: If the tokenizer shape is not recognised.
    """
    if isinstance(tokenizer, (list, tuple)) and len(tokenizer) == 2:
        encode, decode = tokenizer
        return encode, decode

    if hasattr(tokenizer, 'encode') and hasattr(tokenizer, 'decode'):
        def encode(text: str) -> List[int]:
            result = tokenizer.encode(text)
            # tiktoken returns a list; HF tokenizers return a dict or list.
            if isinstance(result, dict):
                return result['input_ids']
            return result

        def decode(ids: List[int]) -> str:
            return tokenizer.decode(ids)

        return encode, decode

    raise TypeError(
        f"Unsupported tokenizer type {type(tokenizer)!r}.  "
        "Pass a tiktoken encoder, a HuggingFace tokenizer, or a "
        "(encode_fn, decode_fn) tuple."
    )


def _load_text_tokens(
    data_path: str,
    encode: Callable[[str], List[int]],
    block_size: int,
) -> torch.Tensor:
    """Load a text file, tokenise it, and return as a LongTensor.

    Args:
        data_path:  Path to a plain-text file.
        encode:     Callable mapping string → list of int token ids.
        block_size: Model context window (for informational reporting).

    Returns:
        1-D LongTensor of token ids.
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    tokens = encode(text)
    print(
        f"[benchmarks] Loaded {len(tokens):,} tokens from {data_path!r} "
        f"(block_size={block_size})"
    )
    return torch.tensor(tokens, dtype=torch.long)


# ---------------------------------------------------------------------------
# LAMBADA evaluation
# ---------------------------------------------------------------------------

def evaluate_lambada(
    model: nn.Module,
    tokenizer,
    data_path: str,
    device: str,
    block_size: int = 1024,
    stride: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate next-word prediction accuracy on the LAMBADA dataset.

    LAMBADA tests a model's ability to predict the final word of a passage
    whose meaning depends on long-range context.  We evaluate by computing
    the probability assigned to the correct last word (or subword sequence),
    treating it as a zero-shot next-token prediction task.

    The ``data_path`` should point to a JSONL file where each line is a JSON
    object with a ``"text"`` key containing the passage.  The last
    whitespace-delimited word is treated as the target.

    If ``data_path`` ends in ``.txt``, one example per line is assumed.

    Args:
        model:      GPT model in eval mode.
        tokenizer:  Tokenizer with ``encode`` / ``decode`` methods or a
                    ``(encode, decode)`` tuple.
        data_path:  Path to LAMBADA data (JSONL or TXT).
        device:     Device string (``'cuda'``, ``'cpu'``, etc.).
        block_size: Model context window size.
        stride:     Sliding-window stride for long contexts (default:
                    ``block_size // 2``).

    Returns:
        Dict with keys:
        - ``'accuracy'``:      Fraction of examples where the last word is
                               the argmax prediction.
        - ``'total_examples'``:Number of evaluated examples.
        - ``'num_correct'``:   Number of correct predictions.
        - ``'mean_loss'``:     Mean per-token cross-entropy over last words.

    Raises:
        FileNotFoundError: If ``data_path`` does not exist.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"LAMBADA data not found at {data_path!r}.")

    encode, decode = _build_encode_decode(tokenizer)
    stride = stride or block_size // 2

    model.eval()

    # Load examples.
    examples: List[str] = []
    ext = os.path.splitext(data_path)[1].lower()
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ext == '.jsonl' or (ext == '' and line.startswith('{')):
                obj = json.loads(line)
                examples.append(obj.get('text', obj.get('passage', '')))
            else:
                examples.append(line)

    num_correct  = 0
    total_loss   = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in examples:
            # Split into context (all but last word) and target (last word).
            parts = text.rsplit(' ', 1)
            if len(parts) < 2:
                continue
            context_text, target_word = parts[0], parts[1]

            context_ids = encode(context_text + ' ')
            target_ids  = encode(target_word)

            if not target_ids:
                continue

            # Build the full token sequence.
            all_ids = context_ids + target_ids
            if len(all_ids) > block_size:
                # Truncate context to fit within block_size.
                all_ids = all_ids[-(block_size):]
                # Recalculate where context ends.
                context_len = len(all_ids) - len(target_ids)
            else:
                context_len = len(context_ids)

            input_ids = torch.tensor(all_ids, dtype=torch.long, device=device).unsqueeze(0)

            # Forward pass over the full sequence.
            logits, _ = model(input_ids)
            # logits: (1, T, vocab_size) — we want predictions at each
            # position in [context_len-1, ..., T-2] to compare against
            # the target_ids tokens.

            # Compute loss and accuracy over the target word tokens only.
            example_correct = True
            for i, tgt_token in enumerate(target_ids):
                pos = context_len - 1 + i   # position whose logit predicts tgt_token
                if pos >= logits.shape[1]:
                    example_correct = False
                    break
                logit_at_pos = logits[0, pos, :]   # (vocab_size,)
                log_probs    = F.log_softmax(logit_at_pos, dim=-1)
                total_loss  += -log_probs[tgt_token].item()
                total_tokens += 1

                predicted_token = logit_at_pos.argmax().item()
                if predicted_token != tgt_token:
                    example_correct = False

            if example_correct:
                num_correct += 1

    n = len(examples)
    accuracy  = num_correct / n if n > 0 else 0.0
    mean_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')

    print(
        f"[LAMBADA] {num_correct}/{n} correct  "
        f"accuracy={accuracy:.4f}  mean_loss={mean_loss:.4f}"
    )
    return {
        'accuracy':       accuracy,
        'total_examples': n,
        'num_correct':    num_correct,
        'mean_loss':      mean_loss,
    }


# ---------------------------------------------------------------------------
# Text completion
# ---------------------------------------------------------------------------

def evaluate_text_completion(
    model: nn.Module,
    prompts: List[str],
    tokenizer,
    device: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: Optional[int] = 200,
    top_p: Optional[float] = None,
    block_size: int = 1024,
) -> List[str]:
    """Generate text completions for a list of prompts.

    Prompts are tokenised, passed through the model's ``generate`` method
    (if available) or through a manual top-k / nucleus sampling loop, and
    the generated tokens are decoded back to strings.

    Args:
        model:          GPT model in eval mode.
        prompts:        List of prompt strings.
        tokenizer:      Tokenizer with ``encode`` / ``decode`` methods or a
                        ``(encode, decode)`` tuple.
        device:         Device string.
        max_new_tokens: Maximum number of tokens to generate per prompt.
        temperature:    Sampling temperature (< 1 = sharper, > 1 = flatter).
        top_k:          Keep only the top-k logits (None = no filtering).
        top_p:          Nucleus probability threshold (None = disabled).
        block_size:     Model context window size.

    Returns:
        List of completion strings (same length as ``prompts``).  Each
        completion is the full text (prompt + generated continuation).
    """
    encode, decode = _build_encode_decode(tokenizer)
    model.eval()

    completions: List[str] = []

    with torch.no_grad():
        for prompt in prompts:
            ids = encode(prompt)
            # Truncate to fit context window.
            if len(ids) > block_size:
                ids = ids[-block_size:]

            x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

            if hasattr(model, 'generate'):
                # Use the model's native generate method.
                y = model.generate(
                    x,
                    max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                generated_ids: List[int] = y[0].tolist()
            else:
                # Manual autoregressive sampling loop.
                generated_ids = ids[:]
                for _ in range(max_new_tokens):
                    ctx_ids = generated_ids[-block_size:]
                    inp = torch.tensor(
                        ctx_ids, dtype=torch.long, device=device
                    ).unsqueeze(0)

                    logits_out = model(inp)
                    if isinstance(logits_out, (tuple, list)):
                        logits = logits_out[0]
                    else:
                        logits = logits_out

                    logits = logits[0, -1, :] / max(temperature, 1e-8)

                    # Top-k filtering.
                    if top_k is not None and top_k > 0:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[-1]] = float('-inf')

                    # Top-p (nucleus) filtering.
                    if top_p is not None and 0.0 < top_p < 1.0:
                        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(
                            F.softmax(sorted_logits, dim=-1), dim=-1
                        )
                        # Remove tokens once cumulative prob exceeds top_p.
                        sorted_logits[cumulative_probs > top_p] = float('-inf')
                        logits = torch.zeros_like(logits).scatter_(
                            0, sorted_idx, sorted_logits
                        )

                    probs     = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                    generated_ids.append(next_token)

            completions.append(decode(generated_ids))

    return completions


# ---------------------------------------------------------------------------
# Perplexity benchmark
# ---------------------------------------------------------------------------

def run_perplexity_benchmark(
    model: nn.Module,
    data_path: str,
    block_size: int,
    device: str,
    tokenizer=None,
    stride: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> float:
    """Compute token-level perplexity on a text file using a sliding window.

    The sliding-window approach (Dagan et al., 2006) avoids boundary effects:
    each forward pass processes ``block_size`` tokens but only the last
    ``stride`` positions contribute to the loss, ensuring every token is
    evaluated with a full (or near-full) context.

    Accepts either:
    - A ``data_path`` pointing to a **pre-tokenised uint16 ``.bin`` file**
      (created by the prepare scripts), or
    - A plain-text ``.txt`` file, in which case a ``tokenizer`` must be given.

    Args:
        model:      GPT model in eval mode.
        data_path:  Path to a ``.bin`` (uint16 token array) or ``.txt`` file.
        block_size: Model context window size.
        device:     Device string.
        tokenizer:  Tokenizer required when ``data_path`` is a text file.
                    Ignored for ``.bin`` files.
        stride:     Sliding-window stride.  Defaults to ``block_size // 2``.
        max_tokens: Limit evaluation to the first ``max_tokens`` tokens.
                    Useful for quick sanity checks.

    Returns:
        Perplexity as a float (lower is better).

    Raises:
        FileNotFoundError: If ``data_path`` does not exist.
        ValueError:        If a text file is given without a tokenizer.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Benchmark data not found at {data_path!r}.")

    stride = stride or block_size // 2

    # Load tokens.
    ext = os.path.splitext(data_path)[1].lower()
    if ext == '.bin':
        tokens_np = np.memmap(data_path, dtype='uint16', mode='r')
        tokens    = torch.from_numpy(tokens_np.astype(np.int64))
    elif ext in ('.txt', '.text', ''):
        if tokenizer is None:
            raise ValueError(
                "A tokenizer must be supplied for plain-text input files."
            )
        encode, _ = _build_encode_decode(tokenizer)
        tokens = _load_text_tokens(data_path, encode, block_size)
    else:
        # Try treating it as binary.
        tokens_np = np.memmap(data_path, dtype='uint16', mode='r')
        tokens    = torch.from_numpy(tokens_np.astype(np.int64))

    if max_tokens is not None:
        tokens = tokens[:max_tokens]

    n = len(tokens)
    if n < 2:
        raise ValueError(
            f"Data at {data_path!r} has only {n} tokens; need at least 2."
        )

    model.eval()

    total_nll    = 0.0   # accumulated negative log-likelihood
    total_tokens = 0     # number of tokens contributing to the NLL

    # Sliding-window evaluation.
    # Each window covers tokens[begin_loc : begin_loc + block_size].
    # The model predicts token t+1 from token t, so for a window of length L
    # we get L-1 next-token predictions.  We use target_ids[begin_loc : end_loc]
    # where end_loc = begin_loc + block_size (capped at n).
    begin_loc = 0
    with torch.no_grad():
        while begin_loc < n - 1:
            end_loc = min(begin_loc + block_size, n)

            # Input: tokens[begin_loc : end_loc]
            input_ids = tokens[begin_loc:end_loc].to(device)
            actual_len = input_ids.shape[0]
            if actual_len < 2:
                break

            # Targets are the next tokens: tokens[begin_loc+1 : end_loc+1].
            # We compute NLL from logits ourselves to avoid shape mismatches
            # that arise when the model internally uses x and y with different
            # lengths at corpus boundaries.
            x = input_ids.unsqueeze(0)  # (1, actual_len)

            # Forward without targets so the model simply returns logits.
            logits_out = model(x)
            if isinstance(logits_out, (tuple, list)):
                logits = logits_out[0]
            else:
                logits = logits_out
            # logits: (1, actual_len, vocab_size)

            # Build target ids: tokens[begin_loc+1 : end_loc+1], capped at n.
            tgt_end    = min(begin_loc + actual_len + 1, n)
            target_ids = tokens[begin_loc + 1 : tgt_end].to(device)
            # Number of valid prediction positions.
            n_preds = min(actual_len - 1, target_ids.shape[0])
            if n_preds < 1:
                break

            # Determine which positions count toward the loss.
            # For the first window, all positions count.
            # For subsequent windows, only the last ``stride`` positions count
            # (to avoid double-counting earlier tokens).
            if begin_loc == 0:
                target_start = 0
            else:
                target_start = max(0, n_preds - stride)

            # logits at position i predicts token i+1, so we use logits[:,:-1,:]
            # paired with target_ids.  Slice to the scored region.
            logits_slice  = logits[0, target_start:n_preds, :]   # (M, V)
            targets_slice = target_ids[target_start:n_preds]      # (M,)

            if logits_slice.shape[0] == 0:
                break

            nll = F.cross_entropy(
                logits_slice, targets_slice, reduction='sum'
            ).item()

            total_nll    += nll
            total_tokens += logits_slice.shape[0]

            next_begin = begin_loc + stride
            if next_begin >= n - 1:
                break
            begin_loc = next_begin

    if total_tokens == 0:
        return float('inf')

    mean_nll    = total_nll / total_tokens
    perplexity  = math.exp(mean_nll)

    print(
        f"[perplexity_benchmark] tokens={total_tokens:,}  "
        f"nll={mean_nll:.4f}  ppl={perplexity:.2f}"
    )
    return perplexity
