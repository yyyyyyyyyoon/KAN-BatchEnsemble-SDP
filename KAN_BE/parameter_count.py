# param_count_ensemble_all.py
import os, sys, glob
import pandas as pd

# 프로젝트 루트 추가(필요시)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from KAN_BE.KAN_BatchEnsemble_model import build_kan_ensemble_model

DEFAULTS = {
    "epochs": 50,
    "n_splits": 10,
    "seed": 42,
    "d_block": 128,
    "n_blocks": 3,
    "degree": 3,
    "k": 4,               # BatchEnsemble 내부 전문가 수(ensemble size)
    "batch_size": 256,
    "lr": 1e-3,
    "d_out": 1,
    "target_col": "class",
}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_input_dim(csv_path, target_col="class"):
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' column not found in {csv_path}")
    return df.drop(columns=[target_col]).shape[1]

if __name__ == "__main__":
    base_dir  = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(base_dir, "..", "data")

    csv_paths = sorted(glob.glob(os.path.join(data_root, "**", "*.csv"), recursive=True))
    if not csv_paths:
        print(f"⚠️ CSV를 찾지 못했습니다: {data_root}")
        sys.exit(0)

    counts = []
    for csv_path in csv_paths:
        dataset = os.path.splitext(os.path.basename(csv_path))[0]
        try:
            input_dim = get_input_dim(csv_path, target_col=DEFAULTS["target_col"])
            model = build_kan_ensemble_model(
                input_dim=input_dim,
                output_dim=DEFAULTS["d_out"],
                k=DEFAULTS["k"],
                d_block=DEFAULTS["d_block"],
                n_blocks=DEFAULTS["n_blocks"],
                degree=DEFAULTS["degree"],
            )
            n_params = count_parameters(model)
            counts.append(n_params)
            print(f"{dataset}: {n_params:,} parameters "
                  f"(k={DEFAULTS['k']}, d_block={DEFAULTS['d_block']}, "
                  f"n_blocks={DEFAULTS['n_blocks']}, degree={DEFAULTS['degree']}, "
                  f"input_dim={input_dim})")
        except Exception as e:
            print(f"❌ {dataset}: 실패 → {e}")

    if counts:
        avg = sum(counts) / len(counts)
        print(f"\n총 {len(counts)}개 데이터셋 평균 파라미터 수: {avg:,.0f}")
