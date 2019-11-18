#!/usr/bin/env bash

python train_adapt.py --gpus 2 --model ../save/xlm_baseline_mix_attn_increasing_schedule --nlayers 12 \
                      --attn_forcing increasing --batch_size 50 --epochs 60 --adapt_epochs 17