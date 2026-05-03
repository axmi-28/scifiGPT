# Single-GPU CUDA SciFi-GPT config.

out_dir = "out-gutenberg-scifi-cuda"
dataset = "gutenberg_scifi"
init_from = "scratch"

eval_interval = 500
log_interval = 10
eval_iters = 200
always_save_checkpoint = True
wandb_log = False

# This preset is still modest compared with GPT-2, but uses a larger context,
# more layers, and gradient accumulation to make better use of a CUDA GPU.
block_size = 256
batch_size = 64
gradient_accumulation_steps = 2

n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.1

learning_rate = 6e-4
max_iters = 10000
lr_decay_iters = 10000
min_lr = 6e-5
beta2 = 0.99
warmup_iters = 500

device = "cuda"
compile = True
