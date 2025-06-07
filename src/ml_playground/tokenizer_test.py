from transformers import AutoTokenizer

#tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
tokenizer = AutoTokenizer.from_pretrained("elyza/ELYZA-japanese-Llama-2-7b-fast")
test_tokenized_ids = tokenizer("Hello, world!")
print(test_tokenized_ids)
test_untokenized = tokenizer.decode(test_tokenized_ids["input_ids"])
print(test_untokenized)
num_ids = tokenizer.vocab_size
print(f"Number of token IDs: {num_ids}")
