from transformers import AutoTokenizer, TextStreamer
from lm_playground.model.qlstm import QLSTMModel, QLSTMConfig
from lm_playground.generator import Generator
import torch

tokenizer = AutoTokenizer.from_pretrained("elyza/ELYZA-japanese-Llama-2-7b-fast", cache_dir="resources/tokenizers")
config = QLSTMConfig(
    dim=1024,
    dim_ff_hidden=2048,
    num_layers=16,
    dropout=0.1,
    vocab_size = tokenizer.vocab_size
)
model = QLSTMModel(config=config)
model.load_state_dict(torch.load("resources/checkpoints/qlstm/qlstm.pth", map_location="cpu"))

# print number of parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params}")

model.eval()
model.cuda()
streamer = TextStreamer(tokenizer, skip_prompt=False, skip_special_tokens=True)
generator = Generator(model, tokenizer, max_length=1024)

while True:
    text_prefix = input("Enter text prefix: ")
    if text_prefix.lower() == "exit":
        break
    generator.generate_stream(text_prefix, streamer)
    generator.reset_hidden()