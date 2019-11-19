#!/usr/bin/env bash

python train_adapt.py --gpus 2 --model ../save/xlm_mix_attn_no_pos_embed --nlayers 12 \
                      --attn_forcing None --batch_size 50 --epochs 60 --adapt_epochs 17 \
                      --pos_embed --resume_train 16