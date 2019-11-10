#!/usr/bin/env bash

python train_adapt.py --gpus 0 --model ../save/xlm_baseline_mix_attn_increase_schedule --nlayers 12 \
                      --attn_forcing increasing --batch_size 40