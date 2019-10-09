#!/usr/bin/env bash

source activate pytorch1dot0
python train.py --dropouti 0.4 --dropouth 0.15 --embd_pdrop 0.1 --model ../save/openai_drop_7
python train.py --dropouti 0.4 --dropouth 0.15 --embd_pdrop 0.2 --model ../save/openai_drop_8
python train.py --dropouti 0.3 --dropouth 0.35 --embd_pdrop 0.1 --model ../save/openai_drop_9
python train.py --dropouti 0.3 --dropouth 0.35 --embd_pdrop 0.2 --model ../save/openai_drop_10
python train.py --dropouti 0.2 --dropouth 0.25 --embd_pdrop 0.2 --model ../save/openai_drop_11
python train.py --dropouti 0.2 --dropouth 0.35 --embd_pdrop 0.2 --model ../save/openai_drop_12
conda deactivate