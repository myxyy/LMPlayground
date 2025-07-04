# MLPlayground

## Overview
myxy's playground for RNN-style language models

## Train example
uv run torchrun --standalone --nproc_per_node=1 src/lm_playground/train_qlstm.py

## Chat example
uv run python src/lm_playground/predict_chat_qlstm.py