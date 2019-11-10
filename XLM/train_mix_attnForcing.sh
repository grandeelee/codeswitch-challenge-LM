#!/usr/bin/env bash

python train_adapt.py --gpus 2 --model ../save/xlm_baseline_mix_attn_constant_schedule --nlayers 12 \
                      --attn_forcing constant --batch_size 40