# count_params_deepensemble.py
import os
import sys
from glob import glob
import pandas as pd
import torch

# 프로젝트 루트를 sys.path에 추가 (import 에러 방지)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from KAN_BE.KAN_BatchEnsemble_model import KAN  # KAN base model

# 기본 하이퍼파라미터(고정값)
DEFAULTS = {
    "ensemble_size": 5,  # ← Deep Ensemble 멤버 수
    "d_block": 128,
    "n_blocks": 3,
    "degree": 3,
    "d_out": 1,          # 이진 분류 출력
    # epochs/lr/batch_size 등은 파라미터 수와 무관
}

def count_parameters(model: torch.nn.Module) -> int:
    """학습되는 파라미터 수만 합산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_input_dim_from_csv(csv_path: str, target_col: str = "class") -> int:
    """CSV에서 'class' 제외 feature 개수를 input_dim으로 사용"""
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' column not found in {csv_path}")
    return df.drop(columns=[target_col]).shape[1]

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(base_dir, "data")

    # 항상 data/**.csv 전부 처리 (인자 무시)
    csv_paths = sorted(glob(os.path.join(data_root, "**", "*.csv"), recursive=True))

    if not csv_paths:
        print(f"⚠️ CSV를 찾지 못했습니다: {data_root}")
        return

    total_counts = []
    ens = DEFAULTS["ensemble_size"]

    for csv_path in csv_paths:
        dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
        try:
            input_dim = get_input_dim_from_csv(csv_path, target_col="class")
        except Exception as e:
            print(f"❌ {dataset_name}: input_dim 추출 실패 → {e}")
            continue

        # 단일 KAN 모델 생성
        model = KAN(
            d_in=input_dim,
            d_out=DEFAULTS["d_out"],
            d_block=DEFAULTS["d_block"],
            n_blocks=DEFAULTS["n_blocks"],
            degree=DEFAULTS["degree"],
        )

        single_params = count_parameters(model)
        total_params = single_params * ens  # Deep Ensemble 총 파라미터

        total_counts.append(total_params)
        print(
            f"{dataset_name}: "
            f"{total_params:,} parameters "
            f"(single={single_params:,}, ensemble_size={ens}, "
            f"input_dim={input_dim}, d_block={DEFAULTS['d_block']}, "
            f"n_blocks={DEFAULTS['n_blocks']}, degree={DEFAULTS['degree']})"
        )

    if total_counts:
        avg_params = sum(total_counts) / len(total_counts)
        print(f"\n총 {len(total_counts)}개 데이터셋 평균 파라미터 수(Deep Ensemble): {avg_params:,.0f}")

if __name__ == "__main__":
    main()
