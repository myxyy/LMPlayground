from datasets import load_dataset
from transformers import AutoTokenizer
from lm_playground.model.qlstm import QLSTMModel, QLSTMConfig
from lm_playground.trainer import Trainer

if __name__ == "__main__":
    dataset = load_dataset("graleo/wikipedia", "20231101.ja", split="train", cache_dir="resources/datasets")
    dataset = dataset["text"]
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
        dataset=dataset,
        max_length=4096,
        max_epochs=1,
        model_name="qlstm",
        checkpoint_path="resources/checkpoints/qlstm",
        checkpoint_interval=5000
    )

    trainer.train()
