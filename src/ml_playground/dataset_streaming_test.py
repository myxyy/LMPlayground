from datasets import load_dataset

#dataset = load_dataset("oscar", "unshuffled_deduplicated_en", split="train", streaming=True)
dataset = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True)
#shuffled_dataset = dataset.shuffle(buffer_size=16384, seed=42)
dataset = dataset.take(10)

print(next(iter(dataset)))
#for data in dataset:
#    print(data["text"])