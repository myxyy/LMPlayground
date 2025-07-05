from transformers import AutoTokenizer
from lm_playground.model.qlstm import QLSTMModel, QLSTMConfig
from lm_playground.generator import Generator
import torch

tokenizer = AutoTokenizer.from_pretrained("elyza/ELYZA-japanese-Llama-2-7b-fast")
config = QLSTMConfig(
    dim=1024,
    dim_ff_hidden=2048,
    num_layers=16,
    dropout=0.1,
    vocab_size = tokenizer.vocab_size
)
model = QLSTMModel(config=config)
model.load_state_dict(torch.load("resources/checkpoints/qlstm/qlstm.pth", map_location="cpu"))
model.eval()
model.cuda()

text_prefix = "仙台市は、"
tokenized_prefix = tokenizer(text_prefix, return_tensors="pt")["input_ids"]

generator = Generator(model, tokenizer, max_length=1024)
text = generator.generate(text_prefix, temperature=1.0)

print(text)
