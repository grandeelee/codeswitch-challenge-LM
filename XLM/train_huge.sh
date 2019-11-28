#!/usr/bin/env bash

python train_huge.py --gpus 2 --model ../save/huge/un_ted_combi --nlayers 12 \
                      --attn_forcing None --batch_size 50 --epochs 60 --adapt_epochs 17 \
                      --data ../data/huge_data/ --max_len 68 --sent_per_epoch 200000