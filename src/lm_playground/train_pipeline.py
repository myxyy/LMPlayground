"""Pipeline-parallel training entry point (model-agnostic).

Usage:
  # Single GPU (falls back to regular single-GPU training)
  uv run torchrun --standalone --nproc_per_node=1 src/lm_playground/train_pipeline.py experiment=qgru

  # 2-stage pipeline
  uv run torchrun --standalone --nproc_per_node=2 src/lm_playground/train_pipeline.py experiment=qlstm

  # 4-stage pipeline
  uv run torchrun --standalone --nproc_per_node=4 src/lm_playground/train_pipeline.py experiment=ttt
"""

from __future__ import annotations

import os

import hydra
import torch
from datasets import concatenate_datasets, load_dataset
from hydra.utils import instantiate
from torch.distributed import destroy_process_group, init_process_group

from lm_playground.pipeline_trainer import PipelineTrainer, SingleGPUTrainer


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def _build_datasets(tokenizer):
    dataset_wiki = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", cache_dir="resources/datasets")
    dataset_wiki_columns = [col for col in dataset_wiki.column_names if col != "text"]
    dataset_wiki = dataset_wiki.remove_columns(dataset_wiki_columns)

    dataset_chat = load_dataset("shi3z/ja_conv_wikipedia_orion14B_100K", split="train", cache_dir="resources/datasets")
    bos = tokenizer.bos_token
    b_inst, e_inst = "[INST]", "[/INST]"
    dataset_chat = dataset_chat.map(
        lambda x: {
            "text": "".join(
                [
                    (bos + b_inst + t["value"] + e_inst if t["from"] == "human" else t["value"])
                    for t in x["conversations"]
                ]
            )
            + bos
        }
    )

    dataset = concatenate_datasets([dataset_wiki, dataset_chat])
    dataset = dataset["text"]
    validation_size = 1000
    train_dataset, validation_dataset = torch.utils.data.random_split(
        dataset,
        [len(dataset) - validation_size, validation_size],
        generator=torch.Generator().manual_seed(42),
    )
    return train_dataset, validation_dataset


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../config/", config_name="config")
def main(cfg):
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    tokenizer = instantiate(cfg.tokenizer.tokenizer)
    train_dataset, validation_dataset = _build_datasets(tokenizer)

    partial_config = instantiate(cfg.model.config)
    config = partial_config(vocab_size=tokenizer.vocab_size)

    if rank == 0:
        print(f"World size: {world_size}")
        print(f"Model config: {config}")

    if world_size == 1:
        # ---- Single GPU: no pipeline ----
        partial_model = instantiate(cfg.model.model)
        model = partial_model(config=config)
        if rank == 0:
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Number of parameters: {num_params:,}")
        partial_optimizer = instantiate(cfg.train.optimizer)
        optimizer = partial_optimizer(params=model.parameters())

        trainer = SingleGPUTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            optimizer=optimizer,
            batch_size=cfg.train.batch_size,
            max_length=cfg.train.max_length,
            max_epochs=cfg.train.max_epochs,
            model_name=cfg.experiment.model_name,
            checkpoint_path=cfg.experiment.checkpoint_path,
            validation_checkpoint_interval=cfg.train.validation_checkpoint_interval,
        )
        trainer.train()
    else:
        # ---- Multi GPU: pipeline parallel ----
        torch.cuda.set_device(rank)
        init_process_group(backend="nccl")

        # Resolve stage class from Hydra config
        partial_stage = instantiate(cfg.model.pipeline_stage)
        stage_class = partial_stage.func

        partial_optimizer = instantiate(cfg.train.optimizer)
        n_microbatches = cfg.train.get("n_microbatches", max(world_size, cfg.train.batch_size))

        trainer = PipelineTrainer(
            stage_class=stage_class,
            config=config,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            optimizer_partial=partial_optimizer,
            batch_size=cfg.train.batch_size,
            max_length=cfg.train.max_length,
            max_epochs=cfg.train.max_epochs,
            model_name=cfg.experiment.model_name,
            checkpoint_path=cfg.experiment.checkpoint_path,
            validation_checkpoint_interval=cfg.train.validation_checkpoint_interval,
            n_microbatches=n_microbatches,
            keep_checkpoints=cfg.train.get("keep_checkpoints", 2),
        )
        trainer.train()

        destroy_process_group()


if __name__ == "__main__":
    main()
