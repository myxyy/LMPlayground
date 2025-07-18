from transformers import TextStreamer
from lm_playground.generator import Generator
import torch
import hydra
from hydra.utils import instantiate

@hydra.main(version_base=None, config_path="../config/", config_name="config")
def main(cfg):
    tokenizer = instantiate(cfg.tokenizer.tokenizer)
    partial_config = instantiate(cfg.model.config)
    config = partial_config(vocab_size = tokenizer.vocab_size)
    partial_model = instantiate(cfg.model.model)
    model = partial_model(config=config)
    model.load_state_dict(torch.load(f"{cfg.experiment.checkpoint_path}/{cfg.experiment.model_name}.pth", map_location="cpu"))

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

if __name__ == "__main__":
    main()