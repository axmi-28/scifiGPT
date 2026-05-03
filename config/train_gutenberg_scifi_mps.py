# Apple Silicon SciFi-GPT config.

out_dir = "out-gutenberg-scifi-mps"
dataset = "gutenberg_scifi"
init_from = "scratch"

eval_interval = 250
log_interval = 10
eval_iters = 100
always_save_checkpoint = True
wandb_log = False

# MPS can handle a longer context and wider model than the CPU preset while
# staying small enough for a local educational training run.
block_size = 256
batch_size = 32
gradient_accumulation_steps = 1

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1

learning_rate = 6e-4
max_iters = 5000
lr_decay_iters = 5000
min_lr = 6e-5
beta2 = 0.99
warmup_iters = 200

device = "mps"
compile = False
dtype = "float32"
