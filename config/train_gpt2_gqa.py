# Train GPT-2 (124M) with Grouped Query Attention extension
#
# Uses 4 KV heads instead of 12, reducing the KV cache memory by 3x while
# maintaining the same number of query heads. Based on Ainslie et al. (2023).
#
# Usage:
#   torchrun --nproc_per_node=8 train.py config/train_gpt2_gqa.py

out_dir = 'out-gpt2-gqa'
eval_interval = 1000
eval_iters = 200
log_interval = 10
wandb_log = True
wandb_project = 'nanogpt'
wandb_run_name = 'gpt2-124M-gqa'

# Same as GPT-2 124M
n_layer = 12
n_head = 12
n_embd = 768
bias = True
dropout = 0.0
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024

# GQA extension - use 4 KV heads instead of 12
attention_type = 'gqa'
n_kv_heads = 4

learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5
lr_schedule = 'cosine'
compile = True
dtype = 'bfloat16'
device = 'auto'
