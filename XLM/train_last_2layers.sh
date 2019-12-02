#!/usr/bin/env bash

python train_last_2layer.py --gpus 0 --model ../save/mix_last_2_layers_multi_target --nlayers 12 \
                      --attn_forcing None --batch_size 50 --epochs 60 --adapt_epochs 17 \
                      --max_len 68 --data ../data/cs_para/