from datasets import load_dataset
from transformers import AutoTokenizer
from ml_playground.model.qlstm import QLSTMLM
from transformers import Trainer, TrainingArguments
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from torch.optim import AdamW
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch

class Trainer:
    def __init__(self, model, tokenizer, dataset, max_length=4096, max_epochs=1):
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_length = max_length
        self.optimizer = AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.01)
        self.max_epochs = max_epochs
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

    def train(self):
        torch.cuda.set_device(self.gpu_id)
        init_process_group(backend="nccl")

        model = DDP(self.model.cuda(), device_ids=[self.gpu_id])
        dataloader = StatefulDataLoader(self.dataset, batch_size=1, pin_memory=True, sampler=DistributedSampler(self.dataset), num_workers=4)

        for epoch in range(self.max_epochs):
            pbar = tqdm(dataloader)
            for i, batch in enumerate(pbar):
                self.optimizer.zero_grad()
                tokenized_text = self.tokenizer(batch["text"], truncation=True, padding="max_length", max_length=self.max_length+1, return_tensors="pt")["input_ids"]
                inputs = tokenized_text[:, :-1].cuda()
                targets = tokenized_text[:, 1:].cuda()
                #print(inputs)
                outputs = model(inputs)
                loss = self.criterion(outputs.view(-1, self.tokenizer.vocab_size), targets.view(-1))
                loss.backward()
            self.optimizer.step()
            pbar.set_postfix({
                "loss": loss.item(),
                "lr": self.optimizer.param_groups[0]["lr"]
            })

        if self.gpu_id == 0:
            if not os.path.exists("resources/weights"):
                os.makedirs("resources/weights")
            torch.save(model.module.cpu().state_dict(), "resources/weights/qlstm.pth")
    
        destroy_process_group()
