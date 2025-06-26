from datasets import load_dataset
from transformers import AutoTokenizer
from lm_playground.model.qlstm import QLSTMModel, QLSTMConfig
from lm_playground.trainer import Trainer
import torch

if __name__ == "__main__":
    dataset = load_dataset("graelo/wikipedia", "20230601.ja", split="train", cache_dir="resources/datasets")
    dataset = dataset["text"]
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [len(dataset) - 1000, 1000], generator=torch.Generator().manual_seed(42))
    #dataset = load_dataset("globis-university/aozorabunko-clean", split="train", cache_dir="resources/datasets")
    #dataset = dataset["text"]
    #dataset = dataset.take(100000)

    tokenizer = AutoTokenizer.from_pretrained("elyza/ELYZA-japanese-Llama-2-7b-fast", cache_dir="resources/tokenizers")

    config = QLSTMConfig(
        dim=1024,
        dim_ff_hidden=2048,
        num_layers=16,
        dropout=0.1
    )
    model = QLSTMModel(
        config=config,
        vocab_size = tokenizer.vocab_size
    )
    model.train()

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        batch_size=6,
        max_length=1024,
        max_epochs=1,
        model_name="qlstm",
        checkpoint_path="resources/checkpoints/qlstm",
        checkpoint_interval=5000,
        validation_interval=500
    )

    trainer.train()
