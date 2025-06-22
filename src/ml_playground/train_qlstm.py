from datasets import load_dataset
from transformers import AutoTokenizer
from ml_playground.model.qlstm import QLSTMLM
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from ml_playground.trainer import Trainer

if __name__ == "__main__":
    dataset = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", cache_dir="resources/datasets")
    dataset = dataset.take(100)

    tokenizer = AutoTokenizer.from_pretrained("elyza/ELYZA-japanese-Llama-2-7b-fast")

    model = QLSTMLM(
        dim = 1024,
        dim_ff_hidden = 2048,
        num_layers = 16,
        dropout = 0.1,
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
        checkpoint_interval=10
    )

    trainer.train()
