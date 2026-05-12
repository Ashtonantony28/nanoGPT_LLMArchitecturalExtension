# Train GPT-2 (124M) on OpenWebText (~2.85 val loss after 300B tokens)
#
# Matches the original GPT-2 124M configuration from Radford et al. (2019).
# Requires ~8 x A100 80 GB GPUs for practical training times.
#
# Usage:
#   torchrun --nproc_per_node=8 train.py config/train_gpt2.py

# I/O
out_dir = 'out-gpt2'
eval_interval = 1000
eval_iters = 200
log_interval = 10
wandb_log = True
wandb_project = 'nanogpt'
wandb_run_name = 'gpt2-124M'

# Data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8  # simulate batch of ~0.5M tokens
batch_size = 12
block_size = 1024

# GPT-2 124M parameters
n_layer = 12
n_head = 12
n_embd = 768
bias = True
dropout = 0.0

# Optimizer
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

# System
device = 'auto'
dtype = 'bfloat16'
compile = True
