#!/usr/bin/env bash

python train_benchmarking.py --gpus 2 --model ../save/cs_benchmarking --nlayers 12 \
                      --attn_forcing None --batch_size 50 --epochs 60 --adapt_epochs 50 \
                      --max_len 68 --data ../data/cs_benchmarking/