import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from dataclasses import dataclass

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight: nn.Parameter | None = (
            nn.Parameter(torch.ones(dim)) if self.elementwise_affine else None
        )

    def forward(self, x: Tensor) -> Tensor:
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            output = output * self.weight
        return output


class FFNSwiGLU(nn.Module):
    def __init__(self, dim: int, dim_ff_hidden: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim_ff_hidden)
        self.fc_act = nn.Linear(dim, dim_ff_hidden)
        self.fc_out = nn.Linear(dim_ff_hidden, dim)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x) * self.act(self.fc_act(x))
        x = self.fc_out(x)
        return x


def scan(a: Tensor, b: Tensor) -> Tensor:
    _, length = a.shape
    if length == 1:
        return b
    is_odd = length % 2 == 1
    a_even = a[:, : -1 if is_odd else None : 2]
    a_odd = a[:, 1::2]
    b_even = b[:, : -1 if is_odd else None : 2]
    b_odd = b[:, 1::2]
    mask_odd = torch.zeros(length, device=a.device, dtype=a.dtype)
    mask_odd[1::2] = 1
    mask_odd = mask_odd[None, :]
    b_new = torch.addcmul(
        torch.addcmul(b, b, mask_odd, value=-1),
        F.pad(
            scan(a_odd * a_even, torch.addcmul(b_odd, a_odd, b_even)).repeat_interleave(
                2, dim=1
            ),
            (0, 1) if is_odd else (0, 0),
            value=0,
        ),
        mask_odd,
    )
    b_odd_new = b_new[:, 1 : None if is_odd else -1 : 2]
    a_even_new = a[:, 2::2]
    mask_even = torch.zeros(length, device=a.device, dtype=a.dtype)
    mask_even[2::2] = 1
    mask_even = mask_even[None, :]
    b_new = torch.addcmul(
        b_new,
        F.pad(
            (a_even_new * b_odd_new).repeat_interleave(2, dim=1),
            (1, 0) if is_odd else (1, 1),
            value=0,
        ),
        mask_even,
    )
    return b_new


class QLSTMLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.fc_forget = nn.Linear(dim, dim)
        self.fc_input = nn.Linear(dim, dim)
        self.fc_input_gate = nn.Linear(dim, dim)
        self.fc_output_gate = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        batch, len, dim = x.shape

        forget = F.sigmoid(self.fc_forget(x))  # (batch, len, dim)

        input = self.tanh(self.fc_input(x)) * self.sigmoid(self.fc_input_gate(x))
        h_inner_chunk = (
            scan(
                forget.transpose(2, 1).reshape(batch * dim, len),
                input.transpose(2, 1).reshape(batch * dim, len),
            )
            .reshape(batch, dim, len)
            .transpose(2, 1)
        )

        h = torch.addcmul(h_inner_chunk, hidden[:, None, :], forget.cumprod(1))

        y = self.tanh(h) * self.sigmoid(self.fc_output_gate(x))
        return y, h[:, -1, :]


class QLSTMBlock(nn.Module):
    def __init__(self, dim: int, dim_ff_hidden: int, dropout: float):
        super().__init__()
        self.qlstm = QLSTMLayer(dim)
        self.ffn = FFNSwiGLU(dim, dim_ff_hidden)
        self.norm_qlstm = RMSNorm(dim)
        self.norm_ffn = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        x_ = x
        x = self.norm_qlstm(x)
        x, hidden = self.qlstm(x, hidden)
        x = self.dropout(x)
        x = x + x_

        x_ = x
        x = self.norm_ffn(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + x_

        return x, hidden

@dataclass
class QLSTMConfig(PretrainedConfig):
    vocab_size: int 
    dim: int = 1024
    dim_ff_hidden: int = 2048
    num_layers: int = 16
    dropout: float = 0.1

class QLSTMPipelineStage(nn.Module):
    """A pipeline-parallel stage containing a subset of QLSTMModel layers."""

    def __init__(
        self,
        config: QLSTMConfig,
        layer_start: int,
        layer_end: int,
        is_first: bool,
        is_last: bool,
    ):
        super().__init__()
        self.is_first = is_first
        self.is_last = is_last
        self.dim = config.dim
        self.num_local_layers = layer_end - layer_start

        if is_first:
            self.embedding = nn.Embedding(config.vocab_size, config.dim)

        self.layers = nn.ModuleList(
            [
                QLSTMBlock(config.dim, config.dim_ff_hidden, config.dropout)
                for _ in range(self.num_local_layers)
            ]
        )

        if is_last:
            self.norm = RMSNorm(config.dim)
            self.fc_out = nn.Linear(config.dim, config.vocab_size)

        self._hidden_init = nn.Parameter(
            torch.zeros(self.num_local_layers, config.dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.is_first:
            x = self.embedding(x)

        batch = x.shape[0]
        hidden = self._hidden_init[None, :].expand(batch, -1, -1)

        for i, layer in enumerate(self.layers):
            x, _ = layer(x, hidden[:, i])

        if self.is_last:
            x = self.norm(x)
            x = self.fc_out(x)

        return x

    @staticmethod
    def split_config(num_layers: int, num_stages: int) -> list[dict]:
        layers_per_stage = num_layers // num_stages
        remainder = num_layers % num_stages
        stages: list[dict] = []
        start = 0
        for i in range(num_stages):
            end = start + layers_per_stage + (1 if i < remainder else 0)
            stages.append(
                {
                    "layer_start": start,
                    "layer_end": end,
                    "is_first": i == 0,
                    "is_last": i == num_stages - 1,
                }
            )
            start = end
        return stages

    def load_from_full_model(self, full_state: dict, layer_start: int) -> None:
        own = {}
        for k, v in full_state.items():
            if self.is_first and k.startswith("embedding."):
                own[k] = v
            if k.startswith("layers."):
                idx = int(k.split(".")[1])
                if layer_start <= idx < layer_start + self.num_local_layers:
                    new_key = f"layers.{idx - layer_start}.{'.'.join(k.split('.')[2:])}"
                    own[new_key] = v
            if self.is_last and (k.startswith("norm.") or k.startswith("fc_out.")):
                own[k] = v
            if k == "_hidden_init":
                own["_hidden_init"] = v[layer_start : layer_start + self.num_local_layers]
        self.load_state_dict(own)

    @staticmethod
    def reconstruct_full_state_dict(gathered: list) -> dict:
        full_sd: dict[str, torch.Tensor] = {}
        hidden_parts: dict[int, torch.Tensor] = {}
        for _rank, info, sd in sorted(gathered, key=lambda x: x[0]):
            layer_start = info["layer_start"]
            for k, v in sd.items():
                v_cpu = v.cpu() if isinstance(v, torch.Tensor) else v
                if k.startswith("embedding."):
                    full_sd[k] = v_cpu
                elif k.startswith("layers."):
                    parts = k.split(".")
                    local_idx = int(parts[1])
                    global_idx = layer_start + local_idx
                    full_sd[f"layers.{global_idx}.{'.'.join(parts[2:])}"] = v_cpu
                elif k.startswith("norm.") or k.startswith("fc_out."):
                    full_sd[k] = v_cpu
                elif k == "_hidden_init":
                    hidden_parts[layer_start] = v_cpu
        if hidden_parts:
            full_sd["_hidden_init"] = torch.cat(
                [hidden_parts[k] for k in sorted(hidden_parts.keys())], dim=0
            )
        return full_sd


class QLSTMModel(PreTrainedModel):
    def __init__(self, config: QLSTMConfig):
        super().__init__(config)
        self.dim = config.dim
        self.num_layers = config.num_layers
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            [QLSTMBlock(config.dim, config.dim_ff_hidden, config.dropout) for _ in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.dim)
        self.fc_out = nn.Linear(config.dim, config.vocab_size)

        self._hidden_init = nn.Parameter(
            torch.zeros(config.num_layers, config.dim)
        )

    def hidden_init(self, batch):
        return self._hidden_init[None, :].expand(batch, -1, -1)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embedding = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.fc_out

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.fc_out = value

    def forward_with_hidden(self, x: Tensor, hidden: Tensor) -> Tensor:
        x = self.embedding(x)
        hidden_next = []
        for i, layer in enumerate(self.layers):
            x, hidden_next_layer = layer(x, hidden[:, i])
            hidden_next.append(hidden_next_layer)
        x = self.norm(x)
        x = self.fc_out(x)
        hidden_next = torch.stack(hidden_next, dim=1)
        return x, hidden_next

    def forward(self, x: Tensor) -> Tensor:
        batch, length = x.shape
        hidden = self.hidden_init(batch)
        hidden = hidden.reshape(batch, self.num_layers, self.dim)

        x, _ = self.forward_with_hidden(x, hidden)
        return x
