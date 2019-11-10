#!/usr/bin/env bash

python train_adapt.py --gpus 3 --model ../save/xlm_baseline_mix_attn_decrease_schedule --nlayers 12 \
                      --attn_forcing decreasing --batch_size 40