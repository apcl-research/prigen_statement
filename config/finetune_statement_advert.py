import time

#out_dir = 'out-owt-gpt2mini'
out_dir = 'out_jam350m-jm_prigen'
eval_interval = 5
eval_iters = 30

wandb_log = True
wandb_project = 'prigen'
wandb_run_name = 'statement-advert'

dataset = 'statement_labels'
init_from = 'resume'

# only save checkpoints if the validation loss improves
always_save_checkpoint = True 

#n_layer = 24
#n_head = 16
#n_embd = 1024
dropout = 0.2

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters

block_size = 1024

batch_size = 4 #16
gradient_accumulation_steps = 32


max_iters = 127000 + 20 * 10 

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
