# parameter_count_kan.py
import os, glob, pandas as pd
from KAN_BE.KAN_BatchEnsemble_model import KAN

DEFAULTS = {"d_block":128, "n_blocks":3, "degree":3, "d_out":1}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_input_dim_from_csv(csv_path, target_col="class"):
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' column not found in {csv_path}")
    return df.drop(columns=[target_col]).shape[1]

def calc_for_csv(csv_path):
    dataset = os.path.splitext(os.path.basename(csv_path))[0]
    input_dim = get_input_dim_from_csv(csv_path)
    model = KAN(
        d_in=input_dim,
        d_out=DEFAULTS["d_out"],
        d_block=DEFAULTS["d_block"],
        n_blocks=DEFAULTS["n_blocks"],
        degree=DEFAULTS["degree"],
    )
    n_params = count_parameters(model)
    print(f"{dataset}: {n_params:,} parameters (input_dim={input_dim})")
    return dataset, n_params

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(base_dir, "data")

    # data/**.csv 전부 수집 (정렬해서 일관된 출력)
    csv_paths = sorted(glob.glob(os.path.join(data_root, "**", "*.csv"), recursive=True))

    if not csv_paths:
        print(f"⚠️ CSV를 찾지 못했습니다: {data_root}")
        raise SystemExit(0)

    counts = []
    for p in csv_paths:
        try:
            _, n = calc_for_csv(p)
            counts.append(n)
        except Exception as e:
            print(f"❌ {p}: {e}")

    if counts:
        avg = sum(counts) / len(counts)
        print(f"\n총 {len(counts)}개 데이터셋 평균 파라미터 수: {avg:,.0f}")
