#!/usr/bin/env bash

python train.py --gpus 3 --model ../save/xlm_interspeech_include_mono --nlayers 12 \
                      --attn_forcing None --batch_size 50 --epochs 60 --adapt_epochs 17 \
                      --max_len 34 --data ../data/cs_para/