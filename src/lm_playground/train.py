from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from lm_playground.model.qgru import QGRUModel, QGRUConfig
from lm_playground.trainer import Trainer
import torch
from schedulefree import RAdamScheduleFree
import os
import hydra
from hydra.utils import instantiate

@hydra.main(version_base=None, config_path="../config/", config_name="config")
def main(cfg):
    tokenizer = AutoTokenizer.from_pretrained("elyza/ELYZA-japanese-Llama-2-7b-fast", cache_dir="resources/tokenizers")

    dataset_wiki = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", cache_dir="resources/datasets")
    dataset_wiki_columns = [col for col in dataset_wiki.column_names if col != "text"]
    dataset_wiki = dataset_wiki.remove_columns(dataset_wiki_columns)
    #dataset = load_dataset("globis-university/aozorabunko-clean", split="train", cache_dir="resources/datasets")
    #dataset = dataset["text"]
    #dataset = dataset.take(100000)

    dataset_chat = load_dataset("shi3z/ja_conv_wikipedia_orion14B_100K", split="train", cache_dir="resources/datasets")
    bos = tokenizer.bos_token
    b_inst, e_inst = "[INST]", "[/INST]"
    #dataset_chat = dataset_chat.map(lambda x : {"text": bos.join([t["value"] for t in x["conversations"]]) + bos})
    dataset_chat = dataset_chat.map(lambda x : {"text": "".join([(bos + b_inst + t["value"] + e_inst if t["from"] == "human" else t["value"]) for t in x["conversations"]]) + bos})

    dataset = concatenate_datasets([dataset_wiki, dataset_chat])
    dataset = dataset["text"]
    validation_size = 1000
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [len(dataset) - validation_size, validation_size], generator=torch.Generator().manual_seed(42))

    partial_config = instantiate(cfg.model.config)
    config = partial_config(vocab_size = tokenizer.vocab_size)
    partial_model = instantiate(cfg.model.model)
    model = partial_model(config=config)
    model.train()

    # print number of parameters
    if int(os.environ["LOCAL_RANK"]) == 0:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters: {num_params}")

    partial_optimizer = instantiate(cfg.train.optimizer)
    optimizer = partial_optimizer(params=model.parameters())

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        batch_size=cfg.train.batch_size,
        max_length=cfg.train.max_length,
        max_epochs=cfg.train.max_epochs,
        model_name=cfg.experiment.model_name,
        checkpoint_path=cfg.experiment.checkpoint_path,
        validation_checkpoint_interval=cfg.train.validation_checkpoint_interval,
        optimizer=optimizer
    )

    trainer.train()

if __name__ == "__main__":
    main()
