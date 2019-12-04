#!/usr/bin/env bash

python train_multi_words_target.py --gpus 0 --model ../save/mix_multi_words_target_2 --nlayers 12 \
                      --attn_forcing None --batch_size 50 --epochs 60 --adapt_epochs 17 \
                      --max_len 68 --data ../data/cs_para/