# Finetune GPT-2 on Shakespeare dataset (start from pretrained weights)
#
# Loads GPT-2 124M pretrained weights from HuggingFace and fine-tunes them on
# the Shakespeare character dataset using LoRA (Low-Rank Adaptation).
#
# LoRA freezes the pretrained weights and injects trainable low-rank matrices,
# reducing the number of trainable parameters from ~124 M to a few million
# while achieving competitive fine-tuning results.
#
# Usage:
#   python train.py config/finetune_shakespeare.py

init_from = 'gpt2'
out_dir = 'out-shakespeare-finetune'
eval_interval = 250
eval_iters = 200

dataset = 'shakespeare_char'
batch_size = 4
gradient_accumulation_steps = 4
block_size = 1024
max_iters = 20000
warmup_iters = 100
lr_decay_iters = 20000
learning_rate = 3e-5
min_lr = 3e-6
decay_lr = True
weight_decay = 0.1

# LoRA finetuning
use_lora = True
lora_rank = 16
lora_alpha = 32.0
