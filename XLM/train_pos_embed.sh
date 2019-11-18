#!/usr/bin/env bash

python train_adapt.py --gpus 3 --model ../save/xlm_mix_attn_constant_pos_embed --nlayers 12 \
                      --attn_forcing constant --batch_size 50 --epochs 60 --adapt_epochs 17 \
                      --pos_embed