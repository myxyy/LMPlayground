from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True)
dataset = dataset.take(10)
tokenizer = AutoTokenizer.from_pretrained("elyza/ELYZA-japanese-Llama-2-7b-fast")
dataset = dataset.map(lambda x: tokenizer(x["text"]))

print(next(iter(dataset))["input_ids"])
print(next(iter(dataset))["input_ids"])

