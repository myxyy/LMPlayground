import torch

class Generator:
    def __init__(self, model, tokenizer, max_length=1024):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._hidden = None

    def reset_hidden(self):
        self._hidden = None

    def generate(self, text_prefix, temperature=1.0):
        if self._hidden is None:
            self._hidden = self.model.hidden_init[None, :, :]
        tokenized_prefix = self.tokenizer(text_prefix, return_tensors="pt")["input_ids"]
        id_list = tokenized_prefix[0].tolist()
        x = tokenized_prefix.to(self._hidden.device)
        for i in range(self.max_length):
            with torch.no_grad():
                y, self._hidden = self.model.forward_with_hidden(x, self._hidden)
            y = y[0, -1, :]
            id_next = torch.multinomial(torch.softmax(y / temperature, dim=-1), num_samples=1)[0]
            id_list.append(id_next.item())
            x = id_next[None, None].to(self._hidden.device)
        text = self.tokenizer.decode(id_list, skip_special_tokens=True)
        return text

    def generate_stream(self, text_prefix, streamer, temperature=1.0, end_token=None):
        if self._hidden is None:
            self._hidden = self.model.hidden_init[None, :, :]
        tokenized_prefix = self.tokenizer(text_prefix, return_tensors="pt")["input_ids"]

        x = tokenized_prefix.cuda() 

        streamer.put(tokenized_prefix)

        max_length = 1024
        for i in range(max_length):
            with torch.no_grad():
                y, self._hidden = self.model.forward_with_hidden(x, self._hidden)
            y = y[0, -1, :]
            id_next = torch.multinomial(torch.softmax(y / temperature, dim=-1), num_samples=1)
            streamer.put(id_next)
            x = id_next[None].cuda()
            if end_token is not None and id_next.item() == end_token:
                break

        streamer.end()



