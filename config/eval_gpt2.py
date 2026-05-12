# Evaluation configuration for GPT-2
#
# Loads the pretrained GPT-2 124M model from HuggingFace and evaluates it
# on the OpenWebText validation set. No training is performed.
#
# Usage:
#   python train.py config/eval_gpt2.py

eval_only = True
init_from = 'gpt2'
dataset = 'openwebtext'
eval_iters = 500
batch_size = 8
block_size = 1024
device = 'auto'
dtype = 'bfloat16'
