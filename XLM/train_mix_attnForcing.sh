#!/usr/bin/env bash

python train_adapt.py --gpus 3 --model ../save/xlm_baseline_mix_attn --nlayers 12 --attn_forcing increasing --batch_size 30