import time
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from options import get_args
from transformer import LMModel
from my_dictionary import Dictionary

args = get_args()

# ================= initialization ========================
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

# ============== data set preparation ======================
contextualized_embeds = []
dico = torch.load('dico.pth')
args.vocab_size = len(dico)
input_test = '请 用 lower case only and 长 度 not longer than 70'
# binarize the input into (batch_size, seqence_length)
input = [dico.index(i) for i in input_test.split()]
input = torch.tensor(input, dtype=torch.long).unsqueeze(0)

# =================== end of data set preparation ============

def get_embed(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    contextualized_embeds.append(output.data)

if __name__ == '__main__':

    model = LMModel(args, args.vocab_size, args.n_ctx)
    model.to(device)
    x = input.to(device)

    model.load_state_dict(torch.load('xlm_baseline_mix_adapt.pt'))
    model.transformer.h._modules['0'].register_forward_hook(get_embed)
    model.transformer.h._modules['1'].register_forward_hook(get_embed)
    model.transformer.h._modules['2'].register_forward_hook(get_embed)
    model.transformer.h._modules['3'].register_forward_hook(get_embed)
    model.transformer.h._modules['4'].register_forward_hook(get_embed)
    model.transformer.h._modules['5'].register_forward_hook(get_embed)
    model.transformer.h._modules['6'].register_forward_hook(get_embed)
    model.transformer.h._modules['7'].register_forward_hook(get_embed)
    model.transformer.h._modules['8'].register_forward_hook(get_embed)
    model.transformer.h._modules['9'].register_forward_hook(get_embed)
    model.transformer.h._modules['10'].register_forward_hook(get_embed)
    model.transformer.h._modules['11'].register_forward_hook(get_embed)

    out = model(x)

    print(x.size())
    print(len(contextualized_embeds), contextualized_embeds[0].size())

    # the embedding for the first word can be a concatenation of all the 12 layers
    first_word = []
    for embed in contextualized_embeds:
        first_word.append(embed[0, 0, :])

