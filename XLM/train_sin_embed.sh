#!/usr/bin/env bash

python train_adapt.py --gpus 0 --model ../save/xlm_mix_attn_no_sin_embed --nlayers 12 \
                      --attn_forcing None --batch_size 50 --epochs 60 --adapt_epochs 17 \
                      --sin_embed