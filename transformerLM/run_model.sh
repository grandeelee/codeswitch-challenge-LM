#!/usr/bin/env bash

source activate pytorch1dot0
python train.py --dropouti 0.4 --dropouth 0.10 --embd_pdrop 0.1 --model ../save/openai_drop_10
python train.py --dropouti 0.4 --dropouth 0.05 --embd_pdrop 0.1 --model ../save/openai_drop_11
python train.py --dropouti 0.4 --dropouth 0.10 --embd_pdrop 0.2 --model ../save/openai_drop_12
conda deactivate