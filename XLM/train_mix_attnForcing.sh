#!/usr/bin/env bash

python train_adapt.py --gpus 1 --model ../save/xlm_baseline_mix_attn --nlayers 12 --attn_forcing decreasing