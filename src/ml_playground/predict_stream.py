from transformers import AutoTokenizer, TextStreamer
from ml_playground.model.qlstm import QLSTMLM
import torch

tokenizer = AutoTokenizer.from_pretrained("elyza/ELYZA-japanese-Llama-2-7b-fast")
model = QLSTMLM(
    dim = 1024,
    dim_ff_hidden = 2048,
    num_layers = 16,
    dropout = 0.1,
    vocab_size = tokenizer.vocab_size
)
model.load_state_dict(torch.load("resources/models/qlstm_lm.pth", map_location="cpu"))
model.eval()
model.cuda()

text_prefix = "仙台市は、"
tokenized_prefix = tokenizer(text_prefix, return_tensors="pt")["input_ids"]

#id_list = tokenized_prefix[0].tolist()

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

x = tokenized_prefix.cuda() 
hidden = model.hidden_init[None,:,:]

streamer.put(tokenized_prefix)

max_length = 1024
for i in range(max_length):
    #print(f"Step {i+1}/{max_length}")
    with torch.no_grad():
        y, hidden = model.forward_with_hidden(x, hidden)
    y = y[0, -1, :]
    #id_next = torch.argmax(y, dim=-1)
    id_next = torch.multinomial(torch.softmax(y, dim=-1), num_samples=1)
    streamer.put(id_next)
    #print(id_next)
    #print(id_next.item())
    #print(id_next.shape)
    #id_list.append(id_next.item())
    x = id_next[None].cuda()

#text = tokenizer.decode(id_list, skip_special_tokens=True)
#print(text)
