from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from lm_playground.model.qlstm import QLSTMModel, QLSTMConfig
from lm_playground.trainer import Trainer
import torch

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("elyza/ELYZA-japanese-Llama-2-7b-fast", cache_dir="resources/tokenizers")

    dataset_wiki = load_dataset("graelo/wikipedia", "20230601.ja", split="train", cache_dir="resources/datasets", trust_remote_code=True)
    dataset_wiki_columns = [col for col in dataset_wiki.column_names if col != "text"]
    dataset_wiki = dataset_wiki.remove_columns(dataset_wiki_columns)
    #dataset = load_dataset("globis-university/aozorabunko-clean", split="train", cache_dir="resources/datasets")
    #dataset = dataset["text"]
    #dataset = dataset.take(100000)

    dataset_chat = load_dataset("shi3z/ja_conv_wikipedia_orion14B_100K", split="train", cache_dir="resources/datasets", trust_remote_code=True)
    bos = tokenizer.bos_token
    b_inst, e_inst = "[INST]", "[/INST]"
    #dataset_chat = dataset_chat.map(lambda x : {"text": bos.join([t["value"] for t in x["conversations"]]) + bos})
    dataset_chat = dataset_chat.map(lambda x : {"text": "".join([(bos + b_inst + t["value"] + e_inst if t["from"] == "human" else t["value"]) for t in x["conversations"]]) + bos})

    dataset = concatenate_datasets([dataset_wiki, dataset_chat])
    dataset = dataset["text"]
    validation_size = 1000
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [len(dataset) - validation_size, validation_size], generator=torch.Generator().manual_seed(42))

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
        validation_checkpoint_interval=500
    )

    trainer.train()
