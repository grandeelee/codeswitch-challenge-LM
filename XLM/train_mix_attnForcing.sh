#!/usr/bin/env bash

python train_adapt.py --gpus 2 --model ../save/xlm_baseline_mix_attn_decrease --nlayers 12 \
                      --attn_forcing decreasing --batch_size 40 --resume_train 1