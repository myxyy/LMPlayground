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


class QGRULayer(nn.Module):
    def __init__(self, dim: int, dim_hidden: int):
        super().__init__()
        self.dim = dim
        self.dim_hidden = dim_hidden
        self.fc_forget = nn.Linear(dim, dim_hidden)
        self.fc_input = nn.Linear(dim, dim_hidden)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.fc_out = nn.Linear(dim_hidden, dim)

    def forward(self, x: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        batch, len, dim = x.shape

        remember = F.sigmoid(self.fc_forget(x)) * torch.linspace(0.0, 1.0, self.dim_hidden, device=x.device)[None, None, :]
        forget = 1 - remember

        input = self.tanh(self.fc_input(x))
        h_inner_chunk = (
            scan(
                forget.transpose(2, 1).reshape(batch * self.dim_hidden, len),
                (input * remember).transpose(2, 1).reshape(batch * self.dim_hidden, len),
            )
            .reshape(batch, self.dim_hidden, len)
            .transpose(2, 1)
        )

        h = torch.addcmul(h_inner_chunk, hidden[:, None, :], forget.cumprod(1))
        y = self.fc_out(h)

        return y, h[:, -1, :]


class QGRUBlock(nn.Module):
    def __init__(self, dim: int, dim_hidden: int, dropout: float):
        super().__init__()
        self.qlstm = QGRULayer(dim, dim_hidden)
        self.ffn = FFNSwiGLU(dim, dim_hidden)
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

class QGRUConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 1024,
        dim_hidden: int = 2048,
        num_layers: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.dropout = dropout

class QGRUPipelineStage(nn.Module):
    """A pipeline-parallel stage containing a subset of QGRUModel layers.

    Each stage owns a contiguous slice of QGRUBlock layers.
    The first stage additionally owns the embedding layer;
    the last stage additionally owns the final norm and output projection.
    """

    def __init__(
        self,
        config: QGRUConfig,
        layer_start: int,
        layer_end: int,
        is_first: bool,
        is_last: bool,
    ):
        super().__init__()
        self.is_first = is_first
        self.is_last = is_last
        self.dim = config.dim
        self.dim_hidden = config.dim_hidden
        self.num_local_layers = layer_end - layer_start

        if is_first:
            self.embedding = nn.Embedding(config.vocab_size, config.dim)

        self.layers = nn.ModuleList(
            [
                QGRUBlock(config.dim, config.dim_hidden, config.dropout)
                for _ in range(self.num_local_layers)
            ]
        )

        if is_last:
            self.norm = RMSNorm(config.dim)
            self.fc_out = nn.Linear(config.dim, config.vocab_size)

        self._hidden_init = nn.Parameter(
            torch.zeros(self.num_local_layers, config.dim_hidden)
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
        """Return split info dicts for *num_stages* pipeline stages."""
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
        """Load weights from a full QGRUModel state_dict into this stage."""
        own = {}
        for k, v in full_state.items():
            if self.is_first and k.startswith("embedding."):
                own[k] = v
            if k.startswith(f"layers."):
                idx = int(k.split(".")[1])
                if layer_start <= idx < layer_start + self.num_local_layers:
                    new_key = f"layers.{idx - layer_start}.{'.'.join(k.split('.')[2:])}"
                    own[new_key] = v
            if self.is_last and (k.startswith("norm.") or k.startswith("fc_out.")):
                own[k] = v
            if k == "_hidden_init":
                own["_hidden_init"] = v[layer_start : layer_start + self.num_local_layers]
        self.load_state_dict(own)


class QGRUModel(PreTrainedModel):
    def __init__(self, config: QGRUConfig):
        super().__init__(config)
        self.dim = config.dim
        self.num_layers = config.num_layers
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            [QGRUBlock(config.dim, config.dim_hidden, config.dropout) for _ in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.dim)
        self.fc_out = nn.Linear(config.dim, config.vocab_size)

        self._hidden_init = nn.Parameter(
            torch.zeros(config.num_layers, config.dim_hidden)
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

        x, _ = self.forward_with_hidden(x, hidden)
        return x
