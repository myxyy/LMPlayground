from datasets import load_dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from torch.optim import AdamW
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from schedulefree import RAdamScheduleFree
from torch.utils.data import DataLoader

class Trainer:
    def __init__(
            self,
            model,
            tokenizer,
            train_dataset,
            validation_dataset,
            model_name,
            batch_size=1,
            max_length=4096,
            max_epochs=1,
            checkpoint_path=None,
            checkpoint_interval=1000,
            validation_interval=1000
        ):
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.batch_size = batch_size
        self.max_length = max_length
        self.optimizer = RAdamScheduleFree(self.model.parameters(), lr=1e-4, betas=(0.9, 0.999))
        self.max_epochs = max_epochs
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.current_step = 0
        self.checkpoint_interval = checkpoint_interval
        self.current_epoch = 0
        self.validation_interval = validation_interval
    
    def save_checkpoint(self):
        if self.gpu_id == 0:
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)

            current_checkpoint_file = None
            checkpoint_files = [f for f in os.listdir(self.checkpoint_path) if f.endswith('.ckpt')]
            if checkpoint_files:
                chackpoint_files_max_epoch = max([int(f.split('_')[2]) for f in checkpoint_files])
                checkpoint_files = [f for f in checkpoint_files if int(f.split('_')[2]) == chackpoint_files_max_epoch]
                latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                current_checkpoint_file = os.path.join(self.checkpoint_path, latest_checkpoint)

            checkpoint_file = os.path.join(self.checkpoint_path, f"{self.model_name}_epoch_{self.current_epoch}_step_{self.current_step}.ckpt")
            torch.save({
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_dataloader_state_dict': self.train_dataloader.state_dict() if hasattr(self, 'train_dataloader') else None,
                'epoch': self.current_epoch,
                'step': self.current_step
            }, checkpoint_file)

            if current_checkpoint_file is not None:
                os.remove(current_checkpoint_file)

            print(f"Checkpoint saved to {checkpoint_file}")

    def load_checkpoint(self):
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            checkpoint_files = [f for f in os.listdir(self.checkpoint_path) if f.endswith('.ckpt')]
            if checkpoint_files:
                chackpoint_files_max_epoch = max([int(f.split('_')[2]) for f in checkpoint_files])
                checkpoint_files = [f for f in checkpoint_files if int(f.split('_')[2]) == chackpoint_files_max_epoch]
                latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                checkpoint_file = os.path.join(self.checkpoint_path, latest_checkpoint)
                checkpoint = torch.load(checkpoint_file, map_location='cpu')
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.train_dataloader.load_state_dict(checkpoint['train_dataloader_state_dict'])
                self.current_epoch = checkpoint['epoch']
                self.current_step = checkpoint['step']
                print(f"Loaded checkpoint from {checkpoint_file}")

    def train(self):
        torch.cuda.set_device(self.gpu_id)
        init_process_group(backend="nccl")

        self.model = DDP(self.model.cuda(), device_ids=[self.gpu_id])
        collator = lambda t: self.tokenizer(t, truncation=True, padding="max_length", max_length=self.max_length+1, return_tensors="pt")
        self.train_dataloader = StatefulDataLoader(self.train_dataset, collate_fn=collator, batch_size=self.batch_size, pin_memory=True, sampler=DistributedSampler(self.train_dataset), num_workers=4)
        self.validation_dataloader = DataLoader(self.validation_dataset, collate_fn=collator, batch_size=self.batch_size, pin_memory=True, sampler=DistributedSampler(self.validation_dataset), num_workers=4)

        self.load_checkpoint()

        while self.current_epoch < self.max_epochs:
            pbar = tqdm(self.train_dataloader, initial=self.current_step, total=len(self.train_dataloader))
            for _, batch in enumerate(pbar):
                self.model.train()
                self.optimizer.train()
                self.optimizer.zero_grad()
                loss = self.calculate_loss(batch)
                loss.backward()
                self.optimizer.step()
                loss_avg = torch.tensor(loss.item(), device=self.gpu_id)
                dist.all_reduce(loss_avg, op=dist.ReduceOp.AVG)
                if self.gpu_id == 0:
                    pbar.set_postfix({
                        "loss": loss_avg.item(),
                        "lr": self.optimizer.param_groups[0]["lr"]
                    })
                self.current_step += 1
                if self.current_step % self.validation_interval == 0:
                    self.model.eval()
                    self.optimizer.eval()
                    pbar_validation = tqdm(self.validation_dataloader)
                    for _, val_batch in enumerate(pbar_validation):
                        with torch.no_grad():
                            val_loss = self.calculate_loss(val_batch)
                            val_loss_avg = torch.tensor(val_loss.item(), device=self.gpu_id)
                            dist.all_reduce(val_loss_avg, op=dist.ReduceOp.AVG)
                            if self.gpu_id == 0:
                                pbar_validation.set_postfix({
                                    "val_loss": val_loss_avg.item()
                                })
                if self.current_step % self.checkpoint_interval == 0:
                    self.save_checkpoint()
            self.current_epoch += 1

        if self.gpu_id == 0:
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            
            weight_file = os.path.join(self.checkpoint_path, f"{self.model_name}.pth")
            torch.save(self.model.module.cpu().state_dict(), weight_file)
    
        destroy_process_group()
    
    def calculate_loss(self, batch):
        inputs = batch["input_ids"][:, :-1].cuda()
        targets = batch["input_ids"][:, 1:].cuda()
        outputs = self.model(inputs)
        loss = self.criterion(outputs.view(-1, self.tokenizer.vocab_size), targets.view(-1))
        return loss
