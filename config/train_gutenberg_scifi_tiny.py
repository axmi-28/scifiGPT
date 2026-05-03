# Tiny from-scratch SciFi-GPT config for CPU laptops.

out_dir = "out-gutenberg-scifi-tiny"
dataset = "gutenberg_scifi"
init_from = "scratch"

eval_interval = 250
log_interval = 10
eval_iters = 50
always_save_checkpoint = True
wandb_log = False

# Short context and small batches keep memory and iteration time manageable.
block_size = 128
batch_size = 16
gradient_accumulation_steps = 1

# A small decoder-only Transformer: token embeddings, causal self-attention,
# MLP blocks, and a language-modeling head trained to predict the next token.
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.1

learning_rate = 6e-4
max_iters = 3000
lr_decay_iters = 3000
min_lr = 6e-5
beta2 = 0.99
warmup_iters = 100

device = "cpu"
compile = False
dtype = "float32"
