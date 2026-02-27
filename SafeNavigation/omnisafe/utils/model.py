# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""This module contains the helper functions for the model."""

from __future__ import annotations

import numpy as np
from torch import nn
import torch
import math

from omnisafe.typing import Activation, InitFunction

class BatchedLinear(nn.Module):
    """Linear layer with an extra leading dimension for ensembles.
    weight: [N, out, in], bias: [N, out]; forward accepts x:[B,in] or [N,B,in] -> [N,B,out].
    """
    def __init__(self, n: int, in_f: int, out_f: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__()
        self.n, self.in_f, self.out_f = n, in_f, out_f
        self.weight = nn.Parameter(torch.empty(n, out_f, in_f, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(n, out_f, device=device, dtype=dtype)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:   # [B,in] -> broadcast to all models
            x = x.unsqueeze(0).expand(self.n, -1, -1)        # [N,B,in]
        y = torch.bmm(x, self.weight.transpose(1, 2))        # [N,B,out]
        if self.bias is not None:
            y = y + self.bias[:, None, :]
        return y


class BatchedMLP(nn.Module):
    """MLP built from BatchedLinear. dims like [Din, H1, ..., Dout].
    Forward: x [B,Din] or [N,B,Din] -> [N,B,Dout].
    """
    def __init__(self, n: int, dims: list[int],
                 activation: type[nn.Module],
                 output_activation: type[nn.Module],
                 relu_inplace: bool = True,
                 bias: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(BatchedLinear(n, dims[i], dims[i + 1], bias=bias, device=device, dtype=dtype))
            # choose act for this position
            if i < len(dims) - 2:
                # ReLU inplace to reduce memory if chosen
                if activation is nn.ReLU and relu_inplace:
                    layers.append(nn.ReLU(inplace=True))
                else:
                    layers.append(activation())
            else:
                if output_activation is nn.ReLU and relu_inplace:
                    layers.append(nn.ReLU(inplace=True))
                else:
                    layers.append(output_activation())
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.layers:
            x = m(x)
        return x


def initialize_layer(init_function: InitFunction, layer: nn.Module) -> None:
    """Initialize the layer with the given initialization function.

    The ``init_function`` can be chosen from: ``kaiming_uniform``, ``xavier_normal``, ``glorot``,
    ``xavier_uniform``, ``orthogonal``.

    Args:
        init_function (InitFunction): The initialization function.
        layer (nn.Linear): The layer to be initialized.
    """
    W = layer.weight
    b = layer.bias

    if init_function == 'kaiming_uniform':
        nn.init.kaiming_uniform_(W, a=math.sqrt(5))
    elif init_function == 'xavier_normal':
        nn.init.xavier_normal_(W)
    elif init_function in ('glorot', 'xavier_uniform'):
        nn.init.xavier_uniform_(W)
    elif init_function == 'orthogonal':
        nn.init.orthogonal_(W, gain=math.sqrt(2))
    else:
        raise TypeError(f'Invalid initialization function: {init_function}')

    if b is not None:
        # mimic nn.Linear default bias init using fan_in
        fan_in = W.size(-1)
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        with torch.no_grad():
            b.uniform_(-bound, bound)


def get_activation(
    activation: Activation,
) -> type[nn.Identity | nn.ReLU | nn.Sigmoid | nn.Softplus | nn.Tanh]:
    """Get the activation function.

    The ``activation`` can be chosen from: ``identity``, ``relu``, ``sigmoid``, ``softplus``,
    ``tanh``.

    Args:
        activation (Activation): The activation function.

    Returns:
        The activation function, ranging from ``nn.Identity``, ``nn.ReLU``, ``nn.Sigmoid``,
        ``nn.Softplus`` to ``nn.Tanh``.
    """
    activations = {
        'identity': nn.Identity,
        'relu': nn.ReLU,
        'sigmoid': nn.Sigmoid,
        'softplus': nn.Softplus,
        'tanh': nn.Tanh,
    }
    assert activation in activations
    return activations[activation]


def build_mlp_network_old(
    sizes: list[int],
    activation: Activation,
    output_activation: Activation = 'identity',
    weight_initialization_mode: InitFunction = 'kaiming_uniform',
) -> nn.Module:
    """Build the MLP network.

    Examples:
        >>> build_mlp_network([64, 64, 64], 'relu', 'tanh')
        Sequential(
            (0): Linear(in_features=64, out_features=64, bias=True)
            (1): ReLU()
            (2): Linear(in_features=64, out_features=64, bias=True)
            (3): ReLU()
            (4): Linear(in_features=64, out_features=64, bias=True)
            (5): Tanh()
        )

    Args:
        sizes (list of int): The sizes of the layers.
        activation (Activation): The activation function.
        output_activation (Activation, optional): The output activation function. Defaults to
            ``identity``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.

    Returns:
        The MLP network.
    """
    activation_fn = get_activation(activation)
    output_activation_fn = get_activation(output_activation)
    layers = []
    for j in range(len(sizes) - 1):
        act_fn = activation_fn if j < len(sizes) - 2 else output_activation_fn
        affine_layer = nn.Linear(sizes[j], sizes[j + 1])
        initialize_layer(weight_initialization_mode, affine_layer)
        layers += [affine_layer, act_fn()]
    return nn.Sequential(*layers)


def build_mlp_network(
    sizes: list[int],
    activation: str,
    output_activation: str = 'identity',
    weight_initialization_mode: str = 'kaiming_uniform',
    *,
    ensemble_size: int = 1,          # NEW: >1 builds a BatchedMLP for N critics
    compile_module: bool = False,     # NEW: optionally wrap with torch.compile
    relu_inplace: bool = True,        # NEW: reduce memory for ReLU
    bias: bool = True,
    device=None,
    dtype=None,
) -> nn.Module:
    """Build an MLP.

    - Default (ensemble_size==1): identical to the original (nn.Sequential of nn.Linear + activations).
    - Batched (ensemble_size>1): returns BatchedMLP running N networks in parallel.
      Forward expects x [B,D] and returns [N,B,Dout] (or [N,B] after your final squeeze).

    Tip: set once at program start for GPU:
        torch.set_float32_matmul_precision('high')
    """
    act_cls = get_activation(activation)
    out_act_cls = get_activation(output_activation)

    if ensemble_size <= 1:
        layers: list[nn.Module] = []
        for j in range(len(sizes) - 1):
            act = act_cls if j < len(sizes) - 2 else out_act_cls
            lin = nn.Linear(sizes[j], sizes[j + 1], bias=bias, device=device, dtype=dtype)
            initialize_layer(weight_initialization_mode, lin)
            # use inplace ReLU if applicable
            if act is nn.ReLU and relu_inplace:
                layers += [lin, nn.ReLU(inplace=True)]
            else:
                layers += [lin, act()]
        net: nn.Module = nn.Sequential(*layers)
    else:
        # Fast fused path: one BatchedMLP replaces N separate MLPs
        net = BatchedMLP(
            n=ensemble_size,
            dims=sizes,
            activation=act_cls,
            output_activation=out_act_cls,
            relu_inplace=relu_inplace,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        # initialize each BatchedLinear with the same schemes
        for m in net.modules():
            if isinstance(m, BatchedLinear):
                initialize_layer(weight_initialization_mode, m)

    if compile_module:
        # Safe compile wrapper (PyTorch 2.x). Works best with static shapes.
        try:
            net = torch.compile(net, fullgraph=True)
        except Exception:
            pass  # silently fall back if not supported
    return net
