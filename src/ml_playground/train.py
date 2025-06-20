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

if __name__ == "__main__":
    gpu_id = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(gpu_id)
    init_process_group(backend="nccl")

    dataset = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", cache_dir="resources/datasets")
    dataset = dataset.take(10000)

    tokenizer = AutoTokenizer.from_pretrained("elyza/ELYZA-japanese-Llama-2-7b-fast")
    #dataset = dataset.map(lambda x: tokenizer(x["text"]))
    dataloader = StatefulDataLoader(dataset, batch_size=1, pin_memory=True, sampler=DistributedSampler(dataset), num_workers=4)

    model = QLSTMLM(
        dim = 1024,
        dim_ff_hidden = 2048,
        num_layers = 16,
        dropout = 0.1,
        vocab_size = tokenizer.vocab_size
    )
    model = DDP(model.cuda(), device_ids=[gpu_id])

    max_length = 4096

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    epochs = 1
    for epoch in range(epochs):
        pbar = tqdm(dataloader)
        for i, batch in enumerate(pbar):
            optimizer.zero_grad()
            tokenized_text = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length+1, return_tensors="pt")["input_ids"]
            inputs = tokenized_text[:, :-1].cuda()
            targets = tokenized_text[:, 1:].cuda()
            #print(inputs)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, tokenizer.vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            pbar.set_postfix({
                "loss": loss.item(),
                "lr": optimizer.param_groups[0]["lr"]
            })
            #print(outputs.shape)  # Should be (batch_size, sequence_length, vocab
    
    destroy_process_group()
