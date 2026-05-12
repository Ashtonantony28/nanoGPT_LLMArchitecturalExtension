# Train a character-level GPT on Shakespeare (~1.47 val loss in ~3 min on A100)
#
# Achieved results (single A100 SXM4 80 GB):
#   ~1.47 validation loss after 5000 iterations
#   ~3 minutes wall-clock time
#
# Usage:
#   python train.py config/train_shakespeare_char.py

# these make the total batch size be ~32K tokens
# 64 sequences of 256 each
batch_size = 64
block_size = 256
gradient_accumulation_steps = 1

# GPT-mini (like the baby gpt in nanoGPT)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# training
max_iters = 5000
lr_decay_iters = 5000
warmup_iters = 100
learning_rate = 1e-3
min_lr = 1e-4
weight_decay = 0.1
decay_lr = True
lr_schedule = 'cosine'

# dataset
dataset = 'shakespeare_char'
init_from = 'scratch'

# logging
eval_interval = 250
eval_iters = 200
log_interval = 10
out_dir = 'out-shakespeare-char'
