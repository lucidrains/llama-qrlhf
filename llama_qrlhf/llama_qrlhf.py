import torch
from torch import nn, einsum, Tensor
from torch.nn import Module

from einops import rearrange, repeat

from beartype import beartype

# helper functions

def exists(v):
    return v is not None

# main classes

class QRLHF(Module):
    @beartype
    def __init__(
        self,
        model: Module
    ):
        super().__init__()

    def forward(self):
        raise NotImplementedError
