# PLAN: nanoGPT Extended (`Ashtonantony28/nanoGPT_LLMArchitecturalExtension`)

Status: **SG0–SG3 built on `plan/nanogpt-fixes`, awaiting review before SG4.**
Base branch: `claude/explore-nanogpt-8L1Rs`. Work branch when approved: `plan/nanogpt-fixes`.
Source of findings: `/mnt/project-files/project-analysis/nanogpt-extended.md` (section numbers cited as §).

## Completeness verdict

**Not complete.** Status is partial.

- Training loop crashes on iteration 0 (`train.py:198`, keyword `schedule=` vs `schedule_type`). Reproduced.
- GPT-2 loading asserts on the override dict and maps weights by position. Not executed; inferred from registration order.
- LoRA trains the full model. Nothing freezes base weights.
- MoE aux loss has no coefficient (adds ≈k per layer).
- OpenWebText prep returns mismatched column lengths. Inferred.
- No inference KV cache, so the GQA inference claim is untestable.
- README and EXTENSIONS benchmark numbers are unmeasured.
- The "153 passing tests" claim was not verified by me: torch is not installed in this environment, and no test runs `train.py`, `sample.py`, or `from_pretrained`.

## Rules applied

- Create this file before any dispatch. Every thread reads it first and updates its own row at the end.
- SG0 (interfaces) merges and is tested before any other build work begins.
- Subagents return only: files changed, tests added, verification command and result, open issues.
- Stop and report after 3 failed verifications, or when a fix needs a shared-interface change.
- Auditors are not the agents that wrote the code.
- Code style: concise, matches existing conventions, minimal comments, no PR bloat.

## Environment

```
python -m pip install torch numpy pytest pyyaml transformers tiktoken
pytest tests -q
```

Install may be limited by the sandbox proxy. Fallback: run the suite in the user's own environment and paste the summary.

## Interface contracts (frozen by SG0)

1. **Model config.** `GPTConfig` fields in `model.py:34-77` are canonical. Every key that `train.py:_build_model_args` (line 268) passes must be a `GPTConfig` field. Enforced by test.
2. **New field:** `GPTConfig.moe_aux_coef: float = 0.01`. Default 0.01. Used by `GPT.forward` (`model.py:265`).
3. **LR schedule factory.** `get_lr_schedule(schedule_type, learning_rate, min_lr, warmup_iters, lr_decay_iters, power=2.0) -> Callable[[int], float]` (`optim/schedules.py:208`). Unchanged. Callers must use this exact signature.
4. **New method:** `GPT.freeze_base() -> None`. Sets `requires_grad=False` on every parameter whose name lacks `lora_`. No-op when `use_lora` is False. Called by `train.py` and by `from_pretrained` consumers.
5. **Checkpoint schema.** Keys: `model`, `optimizer`, `model_args`, `iter_num`, `best_val_loss`, `config`. `model_args` must contain every architecture and extension field, not just the six architecture keys.
6. **Config keys.** Every key in `config/*.py` must be a `train.py` global. Unknown keys raise in `configurator.py`.

## Sub-goals

Dependencies: SG0 first. SG1 runs in parallel with SG0. SG2 gates SG3, SG5, SG10. SG6 gates SG9.

### SG0. Interface freeze (shared, first)
- **Owner:** main thread
- **Status:** done. `tests/test_contracts.py` (5 contract tests), `moe_aux_coef` and `rope_theta` plumbed through `train.py`. Contract 4 (`freeze_base`) lands in SG3.
- **Scope:** implement contracts 1-6 as tests and the `moe_aux_coef` field. No behaviour fixes.
- **Acceptance:**
  - `tests/test_contracts.py` checks that `_build_model_args` keys equal the `GPTConfig` field set.
  - Checkpoint schema test: save and load round-trip keeps `model_args` complete.
  - Config test: every key in `config/*.py` is a known global.
- **Verify:** `pytest tests/test_contracts.py -q`

### SG1. Baseline
- **Owner:** main thread
- **Status:** done. Pre-change suite: 153 passed in 5.9 s on CPU (torch 2.14.1).
- **Scope:** install, run existing suite, record pass count, runtime, and CPU step time for the tiny config.
- **Acceptance:** a baseline file `PLAN-BASELINE.md` with the counts, failures, and timings. Latency budget set here (see below).
- **Verify:** `pytest tests -q --durations=10`

### SG2. Training loop runs
- **Owner:** main thread
- **Status:** done. 50-step CPU run: loss 4.23 → 2.67 by iter 30, no NaN. Also fixed a configurator bug: overrides were silently applied to the importlib frame, so config files and CLI flags were ignored on `import configurator`. Regression test added.
- **Scope:** `train.py:193-240` calls the schedule with the correct keyword and builds it once, outside the loop. Fix the fallback import path.
- **Acceptance:** a 50-step CPU run with `decay_lr=True` completes, with no NaN, and loss at step 50 is lower than at step 0.
- **Verify:**
  ```
  python train.py config/train_shakespeare_char.py --device=cpu --compile=False \
    --max_iters=50 --eval_interval=25 --eval_iters=5 --batch_size=8 --block_size=64 \
    --out_dir=out-sg2
  ```
- **Stop rule:** if a fix requires changing `get_lr_schedule`'s signature, stop (interface change).

### SG3. LoRA freezing
- **Owner:** unassigned
- **Status:** done. `GPT.freeze_base()` in `model.py`, called in `train.py` before `configure_optimizers`. 8192 trainable params for the 2-layer LoRA config (matches Σ r·(in+out)). Base weights bit-identical after 10 steps. Merge equals unmerged within 1e-5. Mutation check: disabling freeze fails 2 tests.
- **Scope:** implement `GPT.freeze_base()` (contract 4). Wire it into `train.py` when `use_lora`.
- **Acceptance:**
  - Trainable parameter count equals Σ over `LoRALinear` of `r·(in+out)`, plus any `lora_`-free parameters that are intentionally trainable (none by default).
  - After 10 optimizer steps, base `weight` tensors are bit-identical to their starting values.
  - `merge()` then forward equals unmerged forward within `1e-5`.
- **Verify:** `pytest tests/test_extensions.py tests/test_training.py -q -k lora`

### SG4. Pretrained GPT-2 loading
- **Owner:** unassigned
- **Status:** not started
- **Scope:** map weights by name (`ff`↔`mlp`, `ln_2`, `attn`), not position (`model.py:396-416`). Allow the override keys that `train.py:342-353` passes, or remove them from that call.
- **Acceptance:** for a fixed token batch, `GPT.from_pretrained('gpt2')` logits match `transformers.GPT2LMHeadModel` logits within `1e-4` max abs diff.
- **Verify:** `python -m pytest tests/test_pretrained.py -q` (new file, skipped if the HF cache is absent, with a note in the output)
- **Needs:** network or a cached `gpt2` checkpoint.

### SG5. Resume carries extension config
- **Owner:** unassigned
- **Status:** not started
- **Scope:** `train.py:326-327` restores all `model_args` keys from the checkpoint, not six.
- **Acceptance:** train 3 steps with `pos_embedding=rope, attention_type=gqa, norm_type=rmsnorm, activation=swiglu, use_moe=True`, save, resume with no extension flags. Loss on the same batch matches to `1e-6`.
- **Verify:** `pytest tests/test_training.py -q -k resume` (new test)

### SG6. MoE correctness
- **Owner:** unassigned
- **Status:** not started
- **Scope:** apply `moe_aux_coef` in `GPT.forward`. Remove the dead `clamp` in `MoEBlock` (`models/extensions.py` ~line 242). Document that `aux` at uniform routing equals `k · coef`.
- **Acceptance:**
  - E=1, k=1 output equals a dense `_ExpertMLP` within `1e-6`.
  - Aux at forced uniform routing equals `k·coef` within `1e-4`.
  - On a synthetic skewed router, one optimizer step reduces the aux loss.
- **Verify:** `pytest tests/test_extensions.py -q -k moe`
- **Note:** changing the default aux scale changes MoE loss values. Record that in the PR.

### SG7. KV cache for generation
- **Owner:** unassigned
- **Status:** not started
- **Scope:** optional per-layer KV cache in `GenerateMixin`-free `GPT.generate`, so decode cost is O(T) per token. Works for MHA and GQA. Cache stores post-RoPE keys, pre-expansion.
- **Acceptance:** cached greedy decode produces identical token ids to uncached decode on a 64-token prompt, 32 new tokens, for MHA and GQA. Decode time per token reported for both.
- **Verify:** `pytest tests/test_model.py -q -k cache`
- **Stop rule:** if equality fails 3 times, stop and report. Do not loosen the check.

### SG8. Data and sample paths
- **Owner:** unassigned
- **Status:** not started
- **Scope:**
  - `data/prepare_openwebtext.py:42-53`: return one row per document, or write with the same batch structure `datasets` expects.
  - `sample.py:122`: find `meta.pkl` in `data/<dataset>/` as well as `out_dir`.
  - `train.py`: copy or reference `meta.pkl` into `out_dir` on checkpoint save.
- **Acceptance:**
  - Prepare script runs on a 1,000-document fixture and produces `train.bin` and `val.bin`.
  - `sample.py --out_dir=out-shakespeare-char` after a short training run decodes characters, not BPE.
- **Verify:** `python data/prepare_openwebtext.py --fixture` (new flag, uses bundled 1,000-line sample) and the sample command above.

### SG9. MoE grouped dispatch (performance)
- **Owner:** unassigned
- **Status:** blocked until SG6 merges
- **Scope:** replace the E×k loop with sort-by-expert and grouped matmul (or `index_add_` per expert), no host sync.
- **Acceptance:** outputs equal the loop version within `1e-5`. Forward and backward time for `B=8, T=256, E=8, k=2` improves; numbers recorded. Not required to beat dense.
- **Verify:** `pytest tests/test_extensions.py -q -k moe` and `python bench.py --profile` on CPU.

### SG10. Docs and config honesty
- **Owner:** unassigned
- **Status:** blocked until SG2 merges
- **Scope:**
  - README and EXTENSIONS: remove or label unmeasured benchmark and "Performance Targets" tables as targets.
  - Correct SwiGLU formula and rounding description.
  - Remove or plumb `rope_theta`, `lora_dropout`, `checkpoint_every_n_layers`.
  - Remove the "FlashAttention" implementation claim, or say it is SDPA dispatch.
- **Acceptance:** every numeric claim in README and EXTENSIONS either links to a command in this repo or is labelled "target".
- **Verify:** SG0 config test passes; manual diff review by an auditor.

### SG11. GPU benchmarks
- **Owner:** user (not runnable in cloud container)
- **Status:** blocked, needs GPU
- **Scope:** `bench.py --compare_attention`, `--compare_precision`, and training runs for the README table.
- **Acceptance:** raw logs committed under `results/`, with hardware and commit hash in each file.

## Latency budget

- CPU unit suite: under 120 s wall time. Confirm in SG1; if baseline is higher, set budget at 1.5× baseline.
- Tiny training step (`block_size=64`, `batch_size=8`, CPU): record in SG1, must not regress more than 20% after any sub-goal.

## Audits

Run after all build sub-goals report. Each writes `AUDIT-<n>.md` at repo root, ranked by severity, each finding with a repro or failing test. Critical and high findings become new sub-goals.

- **AUDIT-1, numerical soundness:** equivalence tests are tight enough to catch wrong math; no test passes with a broken implementation (mutation check: break each fix and confirm its test fails).
- **AUDIT-2, performance:** tok/s and step time for dense vs MoE, generate with vs without cache, memory. Compare to SG1 baseline.
- **AUDIT-3, API and config consistency:** every config key, function signature, and docstring claim matches the code.
- **AUDIT-4, test coverage:** lines covered in `train.py`, `sample.py`, `from_pretrained`, `data/`. Target: every 🔴 item in the analysis has a test that fails before its fix.

Audits repeat until a full session finds no critical or high issues.

## Open questions for the user

1. Approve SG0-SG2 first, or the whole plan?
2. Is a GPU available for SG11, or should it stay out of scope?
3. Should `moe_aux_coef` default to 0.01 (Switch) or 0 (off, keep current behaviour)?
