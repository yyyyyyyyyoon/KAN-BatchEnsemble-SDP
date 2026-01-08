import os, sys, glob
import pandas as pd

import torch
import torch.nn as nn

# 프로젝트 루트 추가(필요시)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiment_sdp.mlp import MLP
from KAN_BE.KAN_BatchEnsemble_model import LinearEfficientEnsemble, make_efficient_ensemble


DEFAULTS = {
    "hidden1": 128,
    "hidden2": 64,
    "k": 5,               # BatchEnsemble 내부 전문가 수
    "d_out": 1,
    "target_col": "class",
}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_input_dim(csv_path: str, target_col: str = "class") -> int:
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' column not found in {csv_path}")
    return df.drop(columns=[target_col]).shape[1]


# ===== BatchEnsemble Wrapper (MLP_BE.py와 동일) =====
class BatchEnsembleWrapper(nn.Module):
    def __init__(self, model: nn.Module, k: int):
        super().__init__()
        self.model = model
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1).expand(-1, self.k, -1)  # (B, K, D)
        B, K, D = x.shape
        x_flat = x.reshape(B * K, D)               # (B*K, D)
        y_flat = self.model(x_flat)                # (B*K, 1)
        y = y_flat.view(B, K, -1)                  # (B, K, 1)
        return y.mean(dim=1)                       # (B, 1)


def build_mlp_be_model(
    input_dim: int,
    output_dim: int,
    k: int,
    hidden1: int,
    hidden2: int,
) -> nn.Module:
    """
    experiment_sdp/MLP_BE.py의 build_mlp_ensemble_model과 동일한 구조로 생성.
    ⚠️ output_dim은 wrapper 구조상 1을 기대하지만, 기록/호환용으로 인자로 유지.
    """
    model = MLP(input_dim=input_dim, hidden1=hidden1, hidden2=hidden2)

    make_efficient_ensemble(
        model,
        LinearEfficientEnsemble,
        k=k,
        ensemble_scaling_in=True,
        ensemble_scaling_out=True,
        ensemble_bias=False,
        scaling_init="random-signs",
    )

    return BatchEnsembleWrapper(model, k=k)


if __name__ == "__main__":
    base_dir  = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(base_dir, "data")

    csv_paths = sorted(glob.glob(os.path.join(data_root, "**", "*.csv"), recursive=True))
    if not csv_paths:
        print(f"⚠️ CSV를 찾지 못했습니다: {data_root}")
        sys.exit(0)

    counts = []
    for csv_path in csv_paths:
        dataset = os.path.splitext(os.path.basename(csv_path))[0]
        try:
            input_dim = get_input_dim(csv_path, target_col=DEFAULTS["target_col"])

            model = build_mlp_be_model(
                input_dim=input_dim,
                output_dim=DEFAULTS["d_out"],
                k=DEFAULTS["k"],
                hidden1=DEFAULTS["hidden1"],
                hidden2=DEFAULTS["hidden2"],
            )

            n_params = count_parameters(model)
            counts.append(n_params)

            print(
                f"{dataset}: {n_params:,} parameters "
                f"(k={DEFAULTS['k']}, hidden1={DEFAULTS['hidden1']}, hidden2={DEFAULTS['hidden2']}, "
                f"input_dim={input_dim})"
            )
        except Exception as e:
            print(f"❌ {dataset}: 실패 → {e}")

    if counts:
        avg = sum(counts) / len(counts)
        print(f"\n총 {len(counts)}개 데이터셋 평균 파라미터 수(MLP-BE): {avg:,.0f}")
