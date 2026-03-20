"""Generic pipeline-parallel and single-GPU trainers.

These trainers are model-agnostic: the caller passes a *stage class*
(e.g. ``QGRUPipelineStage``, ``QLSTMPipelineStage``, ``TTTPipelineStage``)
that knows how to split layers, load from full weights, and reconstruct
a full state dict.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Single-GPU trainer (no pipeline)
# ---------------------------------------------------------------------------

class SingleGPUTrainer:
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        train_dataset,
        validation_dataset,
        optimizer,
        batch_size: int,
        max_length: int,
        max_epochs: int,
        model_name: str,
        checkpoint_path: str,
        validation_checkpoint_interval: int,
    ):
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.cuda(self.gpu_id)
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_epochs = max_epochs
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.validation_checkpoint_interval = validation_checkpoint_interval
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.current_step = 0
        self.current_epoch = 0

    # -- checkpoint helpers ---------------------------------------------------

    def _ckpt_dir(self):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        return self.checkpoint_path

    def save_checkpoint(self):
        d = self._ckpt_dir()
        for f in os.listdir(d):
            if f.endswith(".ckpt"):
                os.remove(os.path.join(d, f))
        path = os.path.join(d, f"{self.model_name}_epoch_{self.current_epoch}_step_{self.current_step}.ckpt")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": self.current_epoch,
                "step": self.current_step,
            },
            path,
        )
        return path

    def save_weight(self):
        d = self._ckpt_dir()
        path = os.path.join(d, f"{self.model_name}.pth")
        torch.save(self.model.state_dict(), path)
        return path

    def load_checkpoint(self):
        d = self.checkpoint_path
        if not d or not os.path.exists(d):
            return
        ckpts = [f for f in os.listdir(d) if f.endswith(".ckpt")]
        if not ckpts:
            return
        latest = max(ckpts, key=lambda x: (int(x.split("_")[2]), int(x.split("_")[-1].split(".")[0])))
        ckpt = torch.load(os.path.join(d, latest), map_location="cpu")
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.current_epoch = ckpt["epoch"]
        self.current_step = ckpt["step"]
        print(f"Loaded checkpoint: {latest}")

    # -- training -------------------------------------------------------------

    def train(self):
        collator = lambda t: self.tokenizer(
            t, truncation=True, padding="max_length", max_length=self.max_length + 1, return_tensors="pt"
        )
        train_dl = DataLoader(
            self.train_dataset, collate_fn=collator, batch_size=self.batch_size, pin_memory=True, shuffle=True, num_workers=4,
        )
        val_dl = DataLoader(
            self.validation_dataset, collate_fn=collator, batch_size=self.batch_size, pin_memory=True, num_workers=4,
        )

        self.load_checkpoint()

        while self.current_epoch < self.max_epochs:
            pbar = tqdm(train_dl, initial=self.current_step, total=len(train_dl), dynamic_ncols=True)
            for batch in pbar:
                self.model.train()
                if hasattr(self.optimizer, "train"):
                    self.optimizer.train()
                self.optimizer.zero_grad()
                inputs = batch["input_ids"][:, :-1].cuda(self.gpu_id)
                targets = batch["input_ids"][:, 1:].cuda(self.gpu_id)
                outputs = self.model(inputs)
                loss = self.criterion(outputs.view(-1, self.tokenizer.vocab_size), targets.view(-1))
                loss.backward()
                self.optimizer.step()
                pbar.set_postfix({"loss": loss.item()})
                self.current_step += 1
                if self.current_step % self.validation_checkpoint_interval == 0:
                    self._validate(val_dl)
                    ckpt = self.save_checkpoint()
                    wt = self.save_weight()
                    tqdm.write(f"Checkpoint: {ckpt}  Weight: {wt}")
            self.save_checkpoint()
            self.save_weight()
            self.current_step = 0
            self.current_epoch += 1

    def _validate(self, val_dl):
        self.model.eval()
        if hasattr(self.optimizer, "eval"):
            self.optimizer.eval()
        pbar = tqdm(val_dl, desc="val")
        for batch in pbar:
            with torch.no_grad():
                inputs = batch["input_ids"][:, :-1].cuda(self.gpu_id)
                targets = batch["input_ids"][:, 1:].cuda(self.gpu_id)
                outputs = self.model(inputs)
                loss = self.criterion(outputs.view(-1, self.tokenizer.vocab_size), targets.view(-1))
                pbar.set_postfix({"val_loss": loss.item()})


# ---------------------------------------------------------------------------
# Pipeline-parallel trainer (model-agnostic)
# ---------------------------------------------------------------------------

class PipelineTrainer:
    """Pipeline-parallel trainer that works with any model's PipelineStage.

    The *stage_class* must implement:
    - ``__init__(config, layer_start, layer_end, is_first, is_last)``
    - ``forward(x) -> Tensor``
    - ``split_config(num_layers, num_stages) -> list[dict]``  (static)
    - ``load_from_full_model(full_state, layer_start)``
    - ``reconstruct_full_state_dict(gathered) -> dict``  (static)
    """

    def __init__(
        self,
        stage_class: type,
        config: Any,
        tokenizer,
        train_dataset,
        validation_dataset,
        optimizer_partial,
        batch_size: int,
        max_length: int,
        max_epochs: int,
        model_name: str,
        checkpoint_path: str,
        validation_checkpoint_interval: int,
        n_microbatches: int = 4,
        keep_checkpoints: int = 2,
    ):
        self.rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.device = torch.device(f"cuda:{self.rank}")
        self.is_first = self.rank == 0
        self.is_last = self.rank == self.world_size - 1

        self.stage_class = stage_class
        self.config = config
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_epochs = max_epochs
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.validation_checkpoint_interval = validation_checkpoint_interval
        self.n_microbatches = n_microbatches
        self.keep_checkpoints = keep_checkpoints
        self.batch_size = max(self.batch_size, self.n_microbatches)
        if self.batch_size != batch_size and self.rank == 0:
            print(f"NOTE: batch_size increased from {batch_size} to {self.batch_size} "
                  f"(must be >= n_microbatches={self.n_microbatches})")
        self.current_step = 0
        self.current_epoch = 0

        # Build this rank's stage module
        stage_infos = stage_class.split_config(config.num_layers, self.world_size)
        self.stage_info = stage_infos[self.rank]
        self.stage_module = stage_class(config, **self.stage_info).to(self.device)

        if self.is_first:
            num_params = sum(p.numel() for p in self.stage_module.parameters())
            for i, info in enumerate(stage_infos):
                print(f"  Stage {i}: layers [{info['layer_start']}..{info['layer_end']})"
                      f"  first={info['is_first']}  last={info['is_last']}")
            print(f"Stage {self.rank} parameters: {num_params:,}")

        self.optimizer = optimizer_partial(params=self.stage_module.parameters())
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

    # -- checkpoint helpers ---------------------------------------------------

    def _ckpt_dir(self):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        return self.checkpoint_path

    def save_checkpoint(self):
        d = self._ckpt_dir()
        path = os.path.join(
            d,
            f"{self.model_name}_stage{self.rank}_epoch_{self.current_epoch}_step_{self.current_step}.ckpt",
        )
        torch.save(
            {
                "stage_state_dict": self.stage_module.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": self.current_epoch,
                "step": self.current_step,
                "stage_info": self.stage_info,
            },
            path,
        )
        self._rotate_checkpoints()
        return path

    def _rotate_checkpoints(self):
        if self.keep_checkpoints <= 0:
            return
        d = self.checkpoint_path
        if not d or not os.path.exists(d):
            return
        prefix = f"{self.model_name}_stage{self.rank}_"
        ckpts = [f for f in os.listdir(d) if f.startswith(prefix) and f.endswith(".ckpt")]
        if len(ckpts) <= self.keep_checkpoints:
            return
        ckpts.sort(key=lambda x: (int(x.split("epoch_")[1].split("_")[0]),
                                   int(x.split("step_")[1].split(".")[0])))
        for old in ckpts[:-self.keep_checkpoints]:
            os.remove(os.path.join(d, old))

    def save_weight(self):
        """Gather all stage weights and reconstruct full model state_dict (rank 0 only)."""
        d = self._ckpt_dir()
        local_sd = self.stage_module.state_dict()
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, (self.rank, self.stage_info, local_sd))
        if not self.is_first:
            return None
        full_sd = self.stage_class.reconstruct_full_state_dict(gathered)
        path = os.path.join(d, f"{self.model_name}.pth")
        torch.save(full_sd, path)
        return path

    def load_checkpoint(self):
        d = self.checkpoint_path
        if not d or not os.path.exists(d):
            return
        # 1) Try pipeline-stage checkpoint first
        prefix = f"{self.model_name}_stage{self.rank}_"
        ckpts = [f for f in os.listdir(d) if f.startswith(prefix) and f.endswith(".ckpt")]
        if ckpts:
            latest = max(ckpts, key=lambda x: (int(x.split("epoch_")[1].split("_")[0]),
                                                int(x.split("step_")[1].split(".")[0])))
            ckpt = torch.load(os.path.join(d, latest), map_location="cpu")
            self.stage_module.load_state_dict(ckpt["stage_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.current_epoch = ckpt["epoch"]
            self.current_step = ckpt["step"]
            if self.is_first:
                print(f"Loaded pipeline checkpoint: {latest}")
            return
        # 2) Fall back to full-model .pth weights
        pth_path = os.path.join(d, f"{self.model_name}.pth")
        if os.path.exists(pth_path):
            full_state = torch.load(pth_path, map_location="cpu")
            self.stage_module.load_from_full_model(full_state, self.stage_info["layer_start"])
            if self.is_first:
                print(f"Loaded full-model weights: {pth_path} (training resumes from epoch 0)")

    # -- pipeline schedule helpers --------------------------------------------

    def _build_pipeline_stage(self):
        from torch.distributed.pipelining import PipelineStage

        return PipelineStage(
            self.stage_module,
            stage_index=self.rank,
            num_stages=self.world_size,
            device=self.device,
        )

    # -- training -------------------------------------------------------------

    def train(self):
        from torch.distributed.pipelining.schedules import ScheduleGPipe

        stage = self._build_pipeline_stage()

        collator = lambda t: self.tokenizer(
            t, truncation=True, padding="max_length", max_length=self.max_length + 1, return_tensors="pt"
        )
        train_sampler = DistributedSampler(self.train_dataset, num_replicas=1, rank=0, shuffle=True)
        train_dl = StatefulDataLoader(
            self.train_dataset,
            collate_fn=collator,
            batch_size=self.batch_size,
            pin_memory=True,
            sampler=train_sampler,
            num_workers=4,
            drop_last=True,
        )
        val_sampler = DistributedSampler(self.validation_dataset, num_replicas=1, rank=0, shuffle=False)
        val_dl = DataLoader(
            self.validation_dataset,
            collate_fn=collator,
            batch_size=self.batch_size,
            pin_memory=True,
            sampler=val_sampler,
            num_workers=4,
            drop_last=True,
        )

        def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return self.criterion(logits.view(-1, self.tokenizer.vocab_size), targets.view(-1))

        schedule = ScheduleGPipe(stage, n_microbatches=self.n_microbatches, loss_fn=loss_fn)

        self.load_checkpoint()

        while self.current_epoch < self.max_epochs:
            train_sampler.set_epoch(self.current_epoch)
            pbar = tqdm(
                train_dl,
                initial=self.current_step,
                total=len(train_dl),
                disable=not self.is_first,
                dynamic_ncols=True,
            )
            for batch in pbar:
                self.stage_module.train()
                if hasattr(self.optimizer, "train"):
                    self.optimizer.train()
                self.optimizer.zero_grad()

                inputs = batch["input_ids"][:, :-1].to(self.device)
                targets = batch["input_ids"][:, 1:].to(self.device)

                if self.is_first:
                    schedule.step(inputs)
                elif self.is_last:
                    losses = schedule.step(target=targets)
                else:
                    schedule.step()

                self.optimizer.step()

                if self.is_last:
                    loss_val = self._compute_loss(losses, loss_fn, targets)
                    loss_tensor = torch.tensor(loss_val, device=self.device)
                else:
                    loss_tensor = torch.tensor(0.0, device=self.device)
                dist.broadcast(loss_tensor, src=self.world_size - 1)
                if self.is_first:
                    pbar.set_postfix({"loss": loss_tensor.item()})

                self.current_step += 1

                if self.current_step % self.validation_checkpoint_interval == 0:
                    self._validate(val_dl, val_sampler, loss_fn)
                    dist.barrier()
                    ckpt = self.save_checkpoint()
                    wt = self.save_weight()
                    if self.is_first:
                        tqdm.write(f"Checkpoint: {ckpt}  Weight: {wt}")

            dist.barrier()
            self.save_checkpoint()
            self.save_weight()
            self.current_step = 0
            self.current_epoch += 1

    def _validate(self, val_dl, val_sampler, loss_fn):
        from torch.distributed.pipelining.schedules import ScheduleGPipe

        self.stage_module.eval()
        if hasattr(self.optimizer, "eval"):
            self.optimizer.eval()

        stage = self._build_pipeline_stage()
        val_schedule = ScheduleGPipe(stage, n_microbatches=self.n_microbatches, loss_fn=loss_fn)

        pbar = tqdm(val_dl, desc="val", disable=not self.is_first)
        for batch in pbar:
            inputs = batch["input_ids"][:, :-1].to(self.device)
            targets = batch["input_ids"][:, 1:].to(self.device)
            if self.is_first:
                val_schedule.step(inputs)
            elif self.is_last:
                outputs = val_schedule.step(target=targets)
            else:
                val_schedule.step()
            if self.is_last:
                with torch.no_grad():
                    loss_val = self._compute_loss(outputs, loss_fn, targets)
                loss_tensor = torch.tensor(loss_val, device=self.device)
            else:
                loss_tensor = torch.tensor(0.0, device=self.device)
            dist.broadcast(loss_tensor, src=self.world_size - 1)
            if self.is_first:
                pbar.set_postfix({"val_loss": loss_tensor.item()})

    @staticmethod
    def _compute_loss(output, loss_fn, targets) -> float:
        if isinstance(output, torch.Tensor):
            with torch.no_grad():
                return loss_fn(output, targets).item()
        if isinstance(output, (list, tuple)):
            mb_sizes = [o.shape[0] for o in output]
            target_mbs = targets.split(mb_sizes)
            total = sum(loss_fn(o, t).item() for o, t in zip(output, target_mbs))
            return total / len(output)
        return 0.0
