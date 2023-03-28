"""Neural network module."""

import math
from typing import Callable, Optional, Sequence, Type, Union

import torch
import torch.distributions as td
import torch.nn as nn

from nanorl import types

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def weight_init_xavier_uniform(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
        activate_final: bool = False,
        use_layer_norm: bool = False,
        dropout_rate: Optional[float] = None,
    ):
        super().__init__()
        all_layer_sizes = [input_dim] + list(hidden_dims)

        _layers = []
        for i, (in_size, out_size) in enumerate(
            zip(all_layer_sizes[:-1], all_layer_sizes[1:])
        ):
            _layers.append(nn.Linear(in_size, out_size))
            if i + 1 < len(hidden_dims) or activate_final:
                if dropout_rate is not None and dropout_rate > 0:
                    _layers.append(nn.Dropout(dropout_rate))
                if use_layer_norm:
                    _layers.append(nn.LayerNorm(out_size))
                _layers.append(activation)

        self._net = nn.Sequential(*_layers)

        # weight initialization
        self.apply(weight_init_xavier_uniform)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x)


class StateActionValue(nn.Module):
    def __init__(
        self,
        base_module: Union[Callable[[], nn.Module], nn.Module],
        module_output_dim: int,
    ):
        super().__init__()
        if isinstance(base_module, nn.Module):
            self._base_module = base_module
        else:
            self._base_module = base_module()
        self.value = nn.Linear(module_output_dim, 1)

        # weight initialization
        self.apply(weight_init_xavier_uniform)

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        inputs = torch.cat([observations, actions], dim=-1)
        value = self.value(self._base_module(inputs))
        return torch.squeeze(value, -1)


class Ensemble(nn.Module):
    def __init__(self, net_cls: Callable[[], nn.Module], num: int = 2):
        super().__init__()
        self._module_ensemble = nn.ModuleList([net_cls() for _ in range(num)])

    def forward(self, *args):
        output = [module(*args) for module in self._module_ensemble]
        return torch.stack(output, dim=0)


class TanhTransform(td.transforms.Transform):
    domain = td.constraints.real
    codomain = td.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x: torch.Tensor):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - nn.functional.softplus(-2.0 * x))


class TanhMultivariateNormalDiag(td.TransformedDistribution):
    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        distribution = td.Normal(loc=loc, scale=scale)
        super().__init__(
            distribution,
            [
                TanhTransform(),
            ],
        )

    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class Normal(nn.Module):
    def __init__(
        self,
        base_module: nn.Module,
        module_output_dim: int,
        action_dim: int,
        log_std_min: float = -20,
        log_std_max: float = 2,
        state_dependent_std: bool = True,
        squash_tanh: bool = False,
    ):
        super().__init__()
        self._base_module = base_module
        self._action_dim = action_dim
        self._log_std_min = log_std_min
        self._log_std_max = log_std_max
        self._state_dependent_std = state_dependent_std
        self._squash_tanh = squash_tanh

        self._means = nn.Linear(module_output_dim, self._action_dim)
        if self._state_dependent_std:
            self._log_stds = nn.Linear(module_output_dim, self._action_dim)
        else:
            self._log_stds = nn.Parameter(
                torch.zeros(self._action_dim), requires_grad=True
            )

        # weight initialization
        self.apply(weight_init_xavier_uniform)

    def forward(self, inputs, *args, **kwargs):
        x = self._base_module(inputs, *args, **kwargs)

        means = self._means(x)
        log_stds = self._log_stds(x)

        log_stds = torch.clamp(log_stds, self._log_std_min, self._log_std_max)
        stds = torch.exp(log_stds)

        if self._squash_tanh:
            return TanhMultivariateNormalDiag(loc=means, scale=stds)
        else:
            return td.MultivariateNormal(loc=means, scale_tril=torch.diag(stds))
