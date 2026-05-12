# EXTENSIONS.md — SOTA Extension Documentation

Detailed guide to each research innovation integrated into nanoGPT Extended, with citations, implementation notes, and benchmarking guidance.

---

## 2.1 Attention Mechanism Improvements

### 2.1.1 Grouped Query Attention (GQA)

**Reference:** "GQA: Training Generalist Multi-Query Transformer Models from Multi-Head Checkpoints" (Ainslie et al., 2023)

**Location:** `models/attention.py` → `GroupedQueryAttention`

**What it does:**  
Standard multi-head attention (MHA) computes a separate Key and Value projection per attention head. GQA groups Q-heads so that multiple Q-heads share a single K/V head. Setting `n_kv_heads = 1` gives Multi-Query Attention (MQA); `n_kv_heads = n_head` recovers standard MHA.

**Implementation:**
```python
# In GPTConfig:
attention_type = 'gqa'   # switch to GQA
n_kv_heads = 4            # 4 KV heads shared among n_head=12 Q-heads

# Internally:
#   q_proj: (n_head * head_dim)
#   kv_proj: (2 * n_kv_heads * head_dim)
#   KV heads are repeated via .repeat_interleave() before attention
```

**How to enable:**
```bash
python train.py config/train_gpt2_gqa.py
# or
python train.py config/train_gpt2.py --attention_type=gqa --n_kv_heads=4
```

**Expected impact:** 20–30% faster inference due to reduced KV cache size.

---

### 2.1.2 Flash Attention

**Reference:** "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (Dao et al., 2023)

**Location:** `models/attention.py` → `CausalSelfAttention`

**What it does:**  
Uses `torch.nn.functional.scaled_dot_product_attention` (PyTorch ≥ 2.0), which dispatches to Flash Attention kernels when available. Falls back to manual attention if not supported.

**Implementation:** Automatic — no configuration needed. The causal mask is passed as `is_causal=True` to enable the Flash Attention path.

**Expected impact:** 2–4× faster attention, significantly less memory for long sequences.

---

### 2.1.3 Rotary Position Embeddings (RoPE)

**Reference:** "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)

**Location:** `models/attention.py` → `RotaryEmbedding`, `apply_rotary_emb`

**What it does:**  
Replaces absolute position embeddings with rotary embeddings applied directly to Q and K inside attention. Rotates pairs of dimensions using position-dependent angles, encoding relative position information.

**Implementation:**
```python
# In GPTConfig:
pos_embedding = 'rope'
rope_theta = 10000.0      # base frequency

# GPT model: no wpe created; RotaryEmbedding used in attention blocks
```

**How to enable:**
```bash
python train.py config/train_gpt2_rope.py
# or
python train.py config/train_gpt2.py --pos_embedding=rope
```

**Expected impact:** Better generalization to longer sequences than learned absolute positions.

---

## 2.2 Activation Functions & Normalization

### 2.2.1 SwiGLU Activation

**Reference:** "GLU Variants Improve Transformer" (Shazeer, 2020)

**Location:** `models/extensions.py` → `SwiGLUMLP`

**What it does:**  
Replaces the standard `GELU(fc1(x))` with a gated variant:
```
gate, up = fc1(x).chunk(2, dim=-1)
output = fc_proj(SiLU(gate) * up)
```

This requires doubling the intermediate dimension of `fc1`, then halving back. The gate mechanism gives the network an efficient multiplicative interaction.

**How to enable:**
```bash
python train.py config/train_gpt2.py --activation=swiglu
```

**Expected impact:** ~2–3% perplexity improvement with the same parameter count.

---

### 2.2.2 RMSNorm

**Reference:** "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)

**Location:** `models/extensions.py` → `RMSNorm`

**What it does:**  
Simplifies LayerNorm by removing the mean-centering step:
```
RMSNorm(x) = x / RMS(x) * weight,   RMS(x) = sqrt(mean(x²) + eps)
```

**How to enable:**
```bash
python train.py config/train_gpt2.py --norm_type=rmsnorm
```

**Expected impact:** 5–10% faster normalization, slightly better convergence.

---

## 2.3 Efficiency & Scaling

### 2.3.1 Low-Rank Adaptation (LoRA)

**Reference:** "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)

**Location:** `models/extensions.py` → `LoRALinear`

**What it does:**  
Replaces weight matrices with `W + (B @ A) * scaling` where `A ∈ ℝ^{rank×in}` and `B ∈ ℝ^{out×rank}`. During fine-tuning, only A and B are updated; the base weight W is frozen.

```python
# LoRALinear:
self.weight   # frozen base weight
self.lora_A   # initialized with Kaiming uniform
self.lora_B   # initialized with zeros (delta = 0 at start)
self.scaling  # = alpha / rank
```

**How to enable:**
```bash
python train.py config/finetune_shakespeare.py  # uses use_lora=True, lora_rank=16
# or
python train.py config/train_gpt2.py --use_lora=True --lora_rank=8 --lora_alpha=16
```

**Save/load LoRA weights only:**
```python
lora_sd = model.get_lora_state_dict()
torch.save(lora_sd, "lora_weights.pt")

model.load_lora_state_dict(torch.load("lora_weights.pt"))
```

**Merge for inference:**
```python
for module in model.modules():
    if isinstance(module, LoRALinear):
        module.merge()   # folds A,B into W for zero-overhead inference
```

**Expected impact:** 10–100× fewer trainable parameters for fine-tuning.

---

### 2.3.2 Mixture of Experts (MoE)

**Reference:** "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" (Fedus et al., 2022)

**Location:** `models/extensions.py` → `MoEBlock`

**What it does:**  
Replaces the feedforward MLP with `num_experts` expert MLPs and a learned router that sends each token to the top-k experts. An auxiliary load-balancing loss encourages uniform expert utilization.

```python
# Router: softmax over expert scores, top-k selection
# Aux loss: encourages uniform expert load
#   aux_loss = num_experts * sum(f_i * P_i)
#   where f_i = fraction of tokens, P_i = mean router probability
```

**How to enable:**
```bash
python train.py config/train_gpt2.py --use_moe=True --num_experts=8 --num_experts_per_tok=2
```

**Expected impact:** 2–3× more parameters with the same computational cost per token.

---

## 2.4 Training & Optimization

### 2.4.1 Learning Rate Schedules

**Location:** `optim/schedules.py`

Available schedules:
- **cosine** (default): cosine decay with linear warmup (matches original nanoGPT)
- **linear**: linear decay with linear warmup
- **constant**: linear warmup then constant LR
- **polynomial**: polynomial decay with configurable power

```bash
python train.py config/train_gpt2.py --lr_schedule=cosine    # default
python train.py config/train_gpt2.py --lr_schedule=linear
python train.py config/train_gpt2.py --lr_schedule=constant
```

---

### 2.4.2 Gradient Checkpointing

**Location:** `model.py` → `GPT.forward`

Selective gradient checkpointing: only checkpoint every `checkpoint_every_n_layers`-th block, giving control over the memory–speed tradeoff.

```bash
python train.py config/train_gpt2.py --gradient_checkpointing=True --checkpoint_every_n_layers=2
```

---

## 2.5 Evaluation Suite

### Perplexity

`eval/metrics.py` → `compute_perplexity(loss)`: simple `exp(loss)`.

### BLEU

`eval/metrics.py` → `compute_bleu(references, hypotheses, n=4)`: corpus-level BLEU-n with clipped n-gram precision and brevity penalty. Pure Python, no external dependencies.

### ROUGE-L

`eval/metrics.py` → `compute_rouge_l(reference, hypothesis)`: Longest Common Subsequence F1 score.

### LAMBADA

`eval/benchmarks.py` → `evaluate_lambada(model, tokenizer, data_path)`: next-word prediction accuracy on the LAMBADA dataset.

### Profiling

`eval/profiler.py` → `profile_inference`, `profile_training_step`, `InferenceProfiler`: measure tokens/sec and peak GPU memory.

```python
from eval.profiler import InferenceProfiler

with InferenceProfiler(model, device="cuda") as prof:
    model.generate(idx, max_new_tokens=100)
prof.print_report()
```

---

## Implementation Philosophy

1. **Modular Design**: Every extension is independently toggleable via `GPTConfig`
2. **Backward Compatibility**: Default config reproduces original nanoGPT exactly
3. **Graceful Degradation**: Extensions have fallbacks (e.g., Flash Attention falls back to standard attention)
4. **Clean Code**: Readability > cleverness; explicit > implicit
5. **No Over-engineering**: Extensions are clean additions, not refactors of the core

---

## Performance Targets

| Metric | Baseline | Extended | Target |
|--------|----------|----------|--------|
| Training Speed (tok/s) | 100K | 150K+ | 150K+ |
| Memory Usage (GB) | 40 | 30 | <40 |
| Inference Speed (tok/s) | 100 | 150+ | 150+ |
| Model Quality (PPL) | 2.85 | 2.75 | 2.75 |
| Fine-tuning Params | 100% | 1–5% (LoRA) | <5% |
