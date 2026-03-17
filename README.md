# MLPlayground

## Overview
myxy's playground for RNN-style language models

## Train example (DDP)
uv run torchrun --standalone --nproc_per_node=1 src/lm_playground/train.py experiment=qgru

## Train example (Pipeline Parallel)
```
# Single GPU
uv run torchrun --standalone --nproc_per_node=1 src/lm_playground/train_pipeline.py experiment=qgru

# Multi GPU (e.g. 2 GPUs)
uv run torchrun --standalone --nproc_per_node=2 src/lm_playground/train_pipeline.py experiment=qgru
```

## Chat example
uv run python src/lm_playground/predict_chat.py experiment=qgru