import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList, RMSNorm
from torch import nn, einsum, Tensor

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

# helpers

def exists(v):
    return v is not None

# rotary

class RotaryEmbedding(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        inv_freq = theta ** -(torch.arange(0, dim, 2).float() / dim)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device = device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        return torch.cat((freqs, freqs), dim = -1)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim = -1)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()

# feedforward

class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

def FeedForward(dim, mult = 4):
    dim_hidden = int(dim * mult * 2 / 3)

    return nn.Sequential(
        RMSNorm(dim),
        nn.Linear(dim, dim_hidden * 2),
        GEGLU(),
        nn.Linear(dim_hidden, dim)
    )

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_hidden = dim_head * heads

        self.to_qkv = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_hidden * 3, bias = False),
            Rearrange('b n (qkv h d) -> qkv b h n d', h = heads, qkv = 3)
        )

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(dim_hidden, dim, bias = False)
        )

    def forward(self, x, rotary_emb = None):
        q, k, v = self.to_qkv(x)

        if exists(rotary_emb):
            q, k = map(lambda t: apply_rotary_pos_emb(rotary_emb, t), (q, k))

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), device = x.device, dtype = torch.bool).triu(j - i + 1)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        return self.to_out(out)

# Q head

class DuelingHead(Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        expansion_factor = 2,
    ):
        super().__init__()
        dim_hidden = int(dim * expansion_factor)

        self.stem = nn.Sequential(
            nn.Linear(dim, dim_hidden),
            nn.SiLU()
        )

        self.to_values = nn.Sequential(
            nn.Linear(dim_hidden, 1)
        )

        self.to_advantages = nn.Sequential(
            nn.Linear(dim_hidden, num_tokens)
        )

    def forward(self, x):
        x = self.stem(x)

        advantages = self.to_advantages(x)
        advantages = advantages - reduce(advantages, '... a -> ... 1', 'mean')

        values = self.to_values(x)

        q_values = values + advantages
        return q_values

# llama

class Llama(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        dueling_q_head = False,
        dueling_q_head_expansion_factor = 2
    ):
        super().__init__()

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.final_norm = RMSNorm(dim)

        self.to_logits = nn.Linear(dim, num_tokens)

        if dueling_q_head:
            self.to_q = DuelingHead(num_tokens = num_tokens, dim = dim, expansion_factor = dueling_q_head_expansion_factor)
        else:
            self.to_q = nn.Linear(dim, num_tokens)

    def forward(
        self,
        x,
        return_q_values = False
    ):
        seq_len, device = x.shape[-1], x.device

        x = self.token_emb(x)

        rotary_emb = self.rotary_emb(seq_len, device = device)

        for attn, ff in self.layers:
            x = attn(x, rotary_emb = rotary_emb) + x
            x = ff(x) + x

        embed = self.final_norm(x)
        logits = self.to_logits(embed)

        if not return_q_values:
            return logits

        return logits, self.to_q(embed)
