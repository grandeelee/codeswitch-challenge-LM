#!/usr/bin/env bash

python train_mono.py --gpus 0 --model ../save/mono_only --nlayers 12 \
                      --attn_forcing None --batch_size 50 --epochs 60 --adapt_epochs 17 \
                      --max_len 68 --data ../data/cs_para/ --resume_train 17