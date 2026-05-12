"""
Evaluation utilities for nanoGPT Extended.

Re-exports the most commonly used symbols from the sub-modules so callers
can write::

    from eval import compute_perplexity, compute_bleu, profile_inference
    from eval import evaluate_lambada, run_perplexity_benchmark
"""

from eval.metrics import (
    compute_perplexity,
    compute_bleu,
    compute_rouge_l,
    compute_token_accuracy,
)

from eval.profiler import (
    profile_inference,
    profile_training_step,
    memory_report,
    InferenceProfiler,
)

from eval.benchmarks import (
    evaluate_lambada,
    evaluate_text_completion,
    run_perplexity_benchmark,
)

__all__ = [
    # Metrics
    'compute_perplexity',
    'compute_bleu',
    'compute_rouge_l',
    'compute_token_accuracy',
    # Profiling
    'profile_inference',
    'profile_training_step',
    'memory_report',
    'InferenceProfiler',
    # Benchmarks
    'evaluate_lambada',
    'evaluate_text_completion',
    'run_perplexity_benchmark',
]
