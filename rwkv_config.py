import json
import os

# dataset path
datafile = os.path.join("data", "enwik8")


# training
lr = 0.00001
epoch = 10
ctx_len = 1024
batch_size = 1
betas = (0.9, 0.999)
eps = 1e-8


import torch
device = "cuda" if torch.cuda.is_available() else "cpu"


# model parm
n_embd = 512
n_layer = 12
