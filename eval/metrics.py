"""
Evaluation metrics for nanoGPT Extended.

Implements perplexity, BLEU-n, ROUGE-L, and token accuracy — all from
scratch with no external NLP library dependencies.

References:
- BLEU: Papineni et al. (2002) https://aclanthology.org/P02-1040/
- ROUGE: Lin (2004) https://aclanthology.org/W04-1013/
"""

from __future__ import annotations

import math
from collections import Counter
from typing import List, Optional, Sequence

import torch


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

def compute_perplexity(loss: float) -> float:
    """Convert a cross-entropy loss (nats) to perplexity.

    Perplexity is defined as ``exp(H)``, where ``H`` is the mean per-token
    cross-entropy.  Lower is better.

    Args:
        loss: Mean per-token cross-entropy loss (in nats, as returned by
              ``torch.nn.functional.cross_entropy``).

    Returns:
        Perplexity value (≥ 1).

    Example::

        ppl = compute_perplexity(3.14)   # ≈ 23.1
    """
    return math.exp(loss)


# ---------------------------------------------------------------------------
# Token accuracy
# ---------------------------------------------------------------------------

def compute_token_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -1,
) -> float:
    """Compute next-token prediction accuracy.

    Args:
        logits:       Float tensor of shape ``(B, T, vocab_size)`` or
                      ``(B*T, vocab_size)``.
        targets:      Long tensor of shape ``(B, T)`` or ``(B*T,)``.
        ignore_index: Token index to mask out (e.g. padding).  Tokens with
                      this target value are excluded from the count.

    Returns:
        Fraction of correct predictions in ``[0, 1]``.

    Example::

        acc = compute_token_accuracy(logits, targets)
        print(f"Accuracy: {acc:.4f}")
    """
    # Flatten to (N, vocab_size) and (N,).
    if logits.dim() == 3:
        B, T, V = logits.shape
        logits  = logits.view(B * T, V)
        targets = targets.view(B * T)

    preds = logits.argmax(dim=-1)

    # Mask positions that should be ignored.
    mask = targets != ignore_index
    if mask.sum() == 0:
        return 0.0

    correct = (preds[mask] == targets[mask]).sum().item()
    total   = mask.sum().item()
    return correct / total


# ---------------------------------------------------------------------------
# BLEU-n
# ---------------------------------------------------------------------------

def _get_ngrams(tokens: List[str], n: int) -> Counter:
    """Return a Counter of n-grams from a token list."""
    return Counter(
        tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)
    )


def _clipped_precision(
    reference_tokens: List[str],
    hypothesis_tokens: List[str],
    n: int,
) -> tuple[int, int]:
    """Compute clipped n-gram precision (numerator, denominator).

    The clipped count of each n-gram in the hypothesis is capped at the
    maximum count of that n-gram in the reference (see BLEU paper §2.1).

    Args:
        reference_tokens:  Tokenised reference string.
        hypothesis_tokens: Tokenised hypothesis string.
        n:                 n-gram order.

    Returns:
        ``(clipped_matches, hypothesis_ngram_count)``
    """
    ref_ngrams  = _get_ngrams(reference_tokens, n)
    hyp_ngrams  = _get_ngrams(hypothesis_tokens, n)

    clipped = 0
    for ngram, count in hyp_ngrams.items():
        clipped += min(count, ref_ngrams.get(ngram, 0))

    return clipped, sum(hyp_ngrams.values())


def compute_bleu(
    references: List[str],
    hypotheses: List[str],
    n: int = 4,
    weights: Optional[Sequence[float]] = None,
    tokenize: bool = True,
) -> float:
    """Corpus-level BLEU-n score.

    Computes the standard BLEU metric from Papineni et al. (2002).  The
    corpus-level score is computed by accumulating clipped n-gram matches
    across all sentence pairs and applying a single brevity penalty.

    This implementation supports BLEU-1 through BLEU-4 (and higher) via the
    ``n`` parameter, with uniform weights by default.  Pass custom ``weights``
    for weighted combinations (must sum to 1).

    Args:
        references:  List of reference strings (one per sample).
        hypotheses:  List of hypothesis / generated strings (one per sample).
        n:           Maximum n-gram order (1–4 typical).
        weights:     Length-``n`` sequence of weights for each n-gram order.
                     Defaults to ``[1/n] * n`` (uniform).
        tokenize:    If True, split on whitespace.  Set to False if inputs are
                     already lists of tokens encoded as space-joined strings.

    Returns:
        BLEU score in ``[0, 1]``.  Returns 0.0 if the hypothesis corpus is
        empty or all n-gram counts are zero.

    Raises:
        ValueError: If ``len(references) != len(hypotheses)`` or weights
                    length does not match ``n``.

    Example::

        refs  = ["the cat is on the mat", "there is a cat on the mat"]
        hyps  = ["the cat is on the mat", "the cat sat on the mat"]
        score = compute_bleu(refs, hyps, n=4)
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"references and hypotheses must have the same length, "
            f"got {len(references)} and {len(hypotheses)}."
        )
    if weights is None:
        weights = [1.0 / n] * n
    if len(weights) != n:
        raise ValueError(f"weights must have length n={n}, got {len(weights)}.")

    if not references:
        return 0.0

    # Accumulate corpus-level statistics.
    clipped_matches  = [0] * n   # numerator for each n-gram order
    hyp_ngram_counts = [0] * n   # denominator for each n-gram order
    ref_length       = 0
    hyp_length       = 0

    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.split() if tokenize else list(ref)
        hyp_tokens = hyp.split() if tokenize else list(hyp)

        ref_length += len(ref_tokens)
        hyp_length += len(hyp_tokens)

        for order in range(1, n + 1):
            matches, total = _clipped_precision(ref_tokens, hyp_tokens, order)
            clipped_matches[order - 1]  += matches
            hyp_ngram_counts[order - 1] += total

    # Brevity penalty: penalise hypotheses shorter than references.
    if hyp_length == 0:
        return 0.0

    bp = 1.0 if hyp_length >= ref_length else math.exp(1.0 - ref_length / hyp_length)

    # Geometric mean of precisions (log-space for numerical stability).
    log_avg_precision = 0.0
    for order in range(n):
        if clipped_matches[order] == 0 or hyp_ngram_counts[order] == 0:
            # Any zero-order precision makes the BLEU score 0.
            return 0.0
        precision = clipped_matches[order] / hyp_ngram_counts[order]
        log_avg_precision += weights[order] * math.log(precision)

    return bp * math.exp(log_avg_precision)


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------

def _lcs_length(a: List[str], b: List[str]) -> int:
    """Compute the length of the Longest Common Subsequence of two token lists.

    Uses the standard DP algorithm: O(|a| * |b|) time, O(min(|a|, |b|)) space
    via the two-row rolling-array optimisation.

    Args:
        a: First token sequence.
        b: Second token sequence.

    Returns:
        LCS length (integer).
    """
    if not a or not b:
        return 0

    # Ensure ``a`` is the shorter sequence for memory efficiency.
    if len(a) > len(b):
        a, b = b, a

    m, n = len(a), len(b)
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)

    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if b[j - 1] == a[i - 1]:
                curr[i] = prev[i - 1] + 1
            else:
                curr[i] = max(curr[i - 1], prev[i])
        prev, curr = curr, [0] * (m + 1)

    return prev[m]


def compute_rouge_l(reference: str, hypothesis: str, tokenize: bool = True) -> float:
    """Sentence-level ROUGE-L F1 score.

    ROUGE-L measures the longest common subsequence (LCS) between the
    reference and hypothesis at the token level.  The F1 formulation balances
    precision (LCS / |hyp|) and recall (LCS / |ref|):

        F1 = (1 + β²) * P * R / (β² * P + R)     with β = 1 (balanced)

    Args:
        reference:  Reference (ground-truth) string.
        hypothesis: Hypothesis (generated) string.
        tokenize:   If True, split on whitespace to get tokens.

    Returns:
        ROUGE-L F1 score in ``[0, 1]``.  Returns 0.0 if either string is
        empty after tokenisation.

    Example::

        score = compute_rouge_l(
            "the cat sat on the mat",
            "the cat sat on a mat",
        )
        # ≈ 0.923
    """
    ref_tokens = reference.split() if tokenize else list(reference)
    hyp_tokens = hypothesis.split() if tokenize else list(hypothesis)

    if not ref_tokens or not hyp_tokens:
        return 0.0

    lcs = _lcs_length(ref_tokens, hyp_tokens)

    precision = lcs / len(hyp_tokens)
    recall    = lcs / len(ref_tokens)

    if precision + recall == 0.0:
        return 0.0

    # F1 with β = 1 (equal weight to precision and recall).
    f1 = 2.0 * precision * recall / (precision + recall)
    return f1
