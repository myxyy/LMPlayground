from transformers import AutoTokenizer, TextStreamer
from lm_playground.model.qlstm import QLSTMModel, QLSTMConfig
import torch

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
model.load_state_dict(torch.load("resources/checkpoints/qlstm/qlstm.pth", map_location="cpu"))

# print number of parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params}")

model.eval()
model.cuda()

def predict(text_prefix):
    tokenized_prefix = tokenizer(text_prefix, return_tensors="pt")["input_ids"]

    streamer = TextStreamer(tokenizer, skip_prompt=False, skip_special_tokens=True)

    x = tokenized_prefix.cuda() 
    hidden = model.hidden_init[None,:,:]

    streamer.put(tokenized_prefix)

    max_length = 1024
    temperature = 1.0
    for i in range(max_length):
        with torch.no_grad():
            y, hidden = model.forward_with_hidden(x, hidden)
        y = y[0, -1, :]
        id_next = torch.multinomial(torch.softmax(y / temperature, dim=-1), num_samples=1)
        streamer.put(id_next)
        x = id_next[None].cuda()

    streamer.end()

while True:
    text_prefix = input("Enter text prefix: ")
    if text_prefix.lower() == "exit":
        break
    predict(text_prefix)