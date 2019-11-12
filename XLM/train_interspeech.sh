#!/usr/bin/env bash

python train.py --gpus 3 --model ../save/xlm_interspeech --nlayers 12 \
                      --attn_forcing None --batch_size 50 --epochs 60 --adapt_epochs 17 \
                      --max_len 68 --data ../data/cs_big/