import itertools
from typing import Any, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, StratifiedKFold

def print_gpu_info():
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU is not available. Using CPU.")

# =====================[ Initialization ]=====================
def init_rsqrt_uniform_(x: Tensor, d: int) -> Tensor:
    assert d > 0
    d_rsqrt = d**-0.5
    return nn.init.uniform_(x, -d_rsqrt, d_rsqrt)


@torch.inference_mode()
def init_random_signs_(x: Tensor) -> Tensor:
    return x.bernoulli_(0.5).mul_(2).add_(-1)


# =====================[ BatchEnsemble Modules ]=====================

class LinearEfficientEnsemble(nn.Module):
    r: None | Tensor
    s: None | Tensor
    bias: None | Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        k: int,
        ensemble_scaling_in: bool,
        ensemble_scaling_out: bool,
        ensemble_bias: bool,
        scaling_init: Literal['ones', 'random-signs'],
    ):
        assert k > 0
        if ensemble_bias:
            assert bias
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_parameter(
            'r',
            (
                nn.Parameter(torch.empty(k, in_features))
                if ensemble_scaling_in
                else None
            ),  # type: ignore[code]
        )
        self.register_parameter(
            's',
            (
                nn.Parameter(torch.empty(k, out_features))
                if ensemble_scaling_out
                else None
            ),  # type: ignore[code]
        )
        self.register_parameter(
            'bias',
            (
                nn.Parameter(torch.empty(out_features))  # type: ignore[code]
                if bias and not ensemble_bias
                else nn.Parameter(torch.empty(k, out_features))
                if ensemble_bias
                else None
            ),
        )

        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.scaling_init = scaling_init
        self.reset_parameters()

    def reset_parameters(self):
        init_rsqrt_uniform_(self.weight, self.in_features)
        scaling_init_fn = {'ones': nn.init.ones_, 'random-signs': init_random_signs_}[
            self.scaling_init
        ]
        if self.r is not None:
            scaling_init_fn(self.r)
        if self.s is not None:
            scaling_init_fn(self.s)
        if self.bias is not None:
            bias_init = torch.empty(
                # NOTE: the shape of bias_init is (out_features,) not (k, out_features).
                # It means that all biases have the same initialization.
                # This is similar to having one shared bias plus
                # k zero-initialized non-shared biases.
                self.out_features,
                dtype=self.weight.dtype,
                device=self.weight.device,
            )
            bias_init = init_rsqrt_uniform_(bias_init, self.in_features)
            with torch.inference_mode():
                self.bias.copy_(bias_init)

    def forward(self, x: Tensor) -> Tensor:
        # x.shape == (B*K, D)
        assert x.ndim == 2

        B_times_K, D = x.shape
        K = self.k
        assert D == self.in_features
        assert B_times_K % K == 0, "입력 배치 크기가 K로 나누어 떨어지지 않습니다"

        B = B_times_K // K

        # (B*K, D) → (B, K, D)
        x = x.view(B, K, D)

        # >>> The equation (5) from the BatchEnsemble paper (arXiv v2).
        if self.r is not None:
            x = x * self.r
        x = x @ self.weight.T
        if self.s is not None:
            x = x * self.s
        # <<<

        if self.bias is not None:
            x = x + self.bias

        x = x.view(B * K, -1)
        return x


def make_efficient_ensemble(module: nn.Module, EnsembleLayer, **kwargs) -> None:
    for name, submodule in list(module.named_children()):
        # Case 1: 일반적인 nn.Linear → 교체
        if isinstance(submodule, nn.Linear):
            module.add_module(
                name,
                EnsembleLayer(
                    in_features=submodule.in_features,
                    out_features=submodule.out_features,
                    bias=submodule.bias is not None,
                    **kwargs,
                ),
            )
        # Case 2: KANLayer → 내부 linear 교체
        elif hasattr(submodule, "linear") and isinstance(getattr(submodule, "linear"), nn.Linear):
            old_linear = submodule.linear
            submodule.linear = EnsembleLayer(
                in_features=old_linear.in_features,
                out_features=old_linear.out_features,
                bias=old_linear.bias is not None,
                **kwargs,
            )
        # Case 3: 다른 서브모듈에 대해 재귀 호출
        else:
            make_efficient_ensemble(submodule, EnsembleLayer, **kwargs)

# =============================[ KAN ]=============================

class KANLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        degree: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree

        # Each input feature will have (degree + 1) learnable basis coefficients
        self.basis_coeffs = nn.Parameter(torch.randn(in_features, degree + 1))  # shape: (in_features, degree + 1)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 3:
            B, K, D = x.shape
            x = x.view(B * K, D)

        basis_list = []
        for d in range(self.degree + 1):
            basis_list.append(x ** d)  # (B, D)

        # Stack and apply learned weights per feature
        # basis_stack: (B, D, degree+1)
        basis_stack = torch.stack(basis_list, dim=-1)

        # Apply basis coefficients: (D, degree+1)
        # result: (B, D)
        projected = (basis_stack * self.basis_coeffs).sum(dim=-1)
        return self.linear(projected)

class KAN(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: Optional[int] = None,
        n_blocks: int = 3,
        d_block: int = 128,
        degree: int = 3,
        dropout: float = 0.0,
        activation: str = "ReLU",
    ):
        super().__init__()
        self.blocks = nn.ModuleList()

        for i in range(n_blocks):
            input_dim = d_in if i == 0 else d_block
            self.blocks.append(KANLayer(input_dim, d_block, degree))
            self.blocks.append(getattr(nn, activation)())
            if dropout > 0.0:
                self.blocks.append(nn.Dropout(dropout))

        self.output = nn.Linear(d_block, d_out) if d_out is not None else None

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.blocks:
            x = layer(x)
        if self.output is not None:
            x = self.output(x)
        return x

class KANBatchEnsembleWrapper(nn.Module): # 입력 확장, 결과 평균
    def __init__(self, model: nn.Module, k: int):
        super().__init__()
        self.model = model  # 공유된 KAN 모델 (BatchEnsemble 교체 완료 상태)
        self.k = k

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, D) → (B, K, D)
        x = x.unsqueeze(1).expand(-1, self.k, -1)

        # Flatten: (B, K, D) → (B*K, D)
        B, K, D = x.shape
        x_flat = x.reshape(B * K, D)

        # Forward pass through shared KAN model
        y_flat = self.model(x_flat)  # (B*K, 1) or (B*K, num_classes)

        # Reshape: (B*K, ...) → (B, K, ...)
        y = y_flat.view(B, K, -1)

        # 평균 앙상블: (B, K, ...) → (B, ...)
        y_mean = y.mean(dim=1)

        return y_mean

def build_kan_ensemble_model(
    input_dim: int,
    output_dim: int,
    k: int,
    d_block: int,
    n_blocks: int,
    degree: int
) -> nn.Module:
    model = KAN(
        d_in=input_dim,
        d_out=output_dim,
        n_blocks=n_blocks,
        d_block=d_block,
        degree=degree
    )
    make_efficient_ensemble(
        model,
        LinearEfficientEnsemble,
        k=k,
        ensemble_scaling_in=True,
        ensemble_scaling_out=True,
        ensemble_bias=False,
        scaling_init="random-signs"
    )
    return KANBatchEnsembleWrapper(model, k=k)


def get_n_parameters(m: nn.Module):
    return sum(x.numel() for x in m.parameters() if x.requires_grad)


@torch.inference_mode()
def compute_parameter_stats(module: nn.Module) -> dict[str, dict[str, float]]:
    stats = {'norm': {}, 'gradnorm': {}, 'gradratio': {}}
    for name, parameter in module.named_parameters():
        stats['norm'][name] = parameter.norm().item()
        if parameter.grad is not None:
            stats['gradnorm'][name] = parameter.grad.norm().item()
            # Avoid computing statistics for zero-initialized parameters.
            if (parameter.abs() > 1e-6).any():
                stats['gradratio'][name] = (
                    (parameter.grad.abs() / parameter.abs().clamp_min_(1e-6))
                    .mean()
                    .item()
                )
    stats['norm']['model'] = (
        torch.cat([x.flatten() for x in module.parameters()]).norm().item()
    )
    stats['gradnorm']['model'] = (
        torch.cat([x.grad.flatten() for x in module.parameters() if x.grad is not None])
        .norm()
        .item()
    )
    return stats


# ======================================================================================
# Optimization
# ======================================================================================
def default_zero_weight_decay_condition(
    module_name: str, module: nn.Module, parameter_name: str, parameter: Parameter
):


    del module_name, parameter
    return parameter_name.endswith('bias') or isinstance(
        module,
        nn.BatchNorm1d
        | nn.LayerNorm
        | nn.InstanceNorm1d
    )


def make_parameter_groups(
    module: nn.Module,
    zero_weight_decay_condition=default_zero_weight_decay_condition,
    custom_groups: None | list[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    if custom_groups is None:
        custom_groups = []
    custom_params = frozenset(
        itertools.chain.from_iterable(group['params'] for group in custom_groups)
    )
    assert len(custom_params) == sum(
        len(group['params']) for group in custom_groups
    ), 'Parameters in custom_groups must not intersect'
    zero_wd_params = frozenset(
        p
        for mn, m in module.named_modules()
        for pn, p in m.named_parameters()
        if p not in custom_params and zero_weight_decay_condition(mn, m, pn, p)
    )
    default_group = {
        'params': [
            p
            for p in module.parameters()
            if p not in custom_params and p not in zero_wd_params
        ]
    }
    return [
        default_group,
        {'params': list(zero_wd_params), 'weight_decay': 0.0},
        *custom_groups,
    ]


def make_optimizer(type: str, **kwargs) -> torch.optim.Optimizer:
    Optimizer = getattr(torch.optim, type)
    return Optimizer(**kwargs)



