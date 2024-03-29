#!/usr/bin/env bash

python train_adapt.py --gpus 0 --model ../save/xlm_baseline_mix_attn_decreasing_schedule --nlayers 12 \
                      --attn_forcing decreasing --batch_size 50 --epochs 60 --adapt_epochs 17 --resume_train 9