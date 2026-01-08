import glob
import os
import sys
import time
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from KAN_BE.KAN_BatchEnsemble_model import KAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA 사용 가능:", torch.cuda.is_available())
print("GPU 이름:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

DEFAULTS = {
    "epochs": 50,
    "n_splits": 10,
    "seed": 2026,
    "d_block": 128,
    "n_blocks": 3,
    "degree": 3,
    "lr": 1e-3,
    "batch_size": 128,
    "ensemble_size": 5,
}

def evaluate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TP = cm[1, 1] if cm.shape == (2, 2) else 0
    FN = cm[1, 0] if cm.shape == (2, 2) else 0
    FP = cm[0, 1] if cm.shape == (2, 2) else 0
    TN = cm[0, 0] if cm.shape == (2, 2) else 0
    PD = TP / (TP + FN + 1e-6)
    PF = FP / (FP + TN + 1e-6)
    FIR = TP / (TP + FP + 1e-6)
    Balance = 1 - (np.sqrt((1 - PD) ** 2 + PF ** 2) / np.sqrt(2))
    return {"PD": PD, "PF": PF, "FIR": FIR, "Balance": Balance}

def get_all_csv_paths(root_dir: str):
    return sorted(glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True))

def load_xy(csv_path: str, target_col: str = "class"):
    import pandas as pd
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' 컬럼을 찾을 수 없습니다. ({csv_path})")
    X = df.drop(columns=[target_col]).to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy().astype(np.int64)
    return X, y

def run_deep_ensemble_final(
    csv_path: str,
    *,
    epochs=DEFAULTS["epochs"],
    n_splits=DEFAULTS["n_splits"],
    seed=DEFAULTS["seed"],
    d_block=DEFAULTS["d_block"],
    n_blocks=DEFAULTS["n_blocks"],
    degree=DEFAULTS["degree"],
    lr=DEFAULTS["lr"],
    batch_size=DEFAULTS["batch_size"],
    ensemble_size=DEFAULTS["ensemble_size"],
):
    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
    print(f"\n===== [{dataset_name}] =====")

    X_all, y_all = load_xy(csv_path)
    input_dim = X_all.shape[1]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all), start=1):
        print(f"\n[Fold {fold}/{n_splits}]")

        # Split
        X_tr, X_te = X_all[train_idx], X_all[test_idx]
        y_tr, y_te = y_all[train_idx], y_all[test_idx]

        # Scale (train fit → test transform)
        scaler = MinMaxScaler()
        X_tr_scaled = scaler.fit_transform(X_tr).astype(np.float32)
        X_te_scaled = scaler.transform(X_te).astype(np.float32)

        # SMOTE (train only)
        smote = SMOTE(random_state=seed)
        try:
            X_tr_res, y_tr_res = smote.fit_resample(X_tr_scaled, y_tr)
        except ValueError as e:
            print(f"[WARN] SMOTE 실패: {e} -> 원본 train 사용")
            X_tr_res, y_tr_res = X_tr_scaled, y_tr

        # Tensor & DataLoader
        X_tr_t = torch.from_numpy(X_tr_res).float()
        y_tr_t = torch.from_numpy(y_tr_res.astype(np.float32))
        X_te_t = torch.from_numpy(X_te_scaled).float()
        y_te_t = torch.from_numpy(y_te.astype(np.float32))

        train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=batch_size, shuffle=False)

        # ===== Ensemble 학습 =====
        models: List[nn.Module] = []
        for _ in range(ensemble_size):
            model = KAN(d_in=input_dim, d_out=1, d_block=d_block, n_blocks=n_blocks, degree=degree).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.BCEWithLogitsLoss()

            model.train()
            for _ in range(epochs):
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    logits = model(xb).view(-1)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()

            models.append(model)

        # ===== 앙상블 엔드투엔드 추론시간 측정 (ms 단위) =====
        for m in models:
            m.eval()
        torch.backends.cudnn.benchmark = True

        preds_list, trues_list = [], []
        with torch.inference_mode():
            for xb, _ in test_loader:
                xb = xb.to(device, non_blocking=True)
                for m in models:
                    _ = m(xb)
                break

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                starter.record()

                for xb, yb in test_loader:
                    xb = xb.to(device, non_blocking=True)
                    probs_accum = None
                    for m in models:
                        logits = m(xb).view(-1)
                        probs = torch.sigmoid(logits)
                        probs_accum = probs if probs_accum is None else probs_accum + probs
                    probs_mean = (probs_accum / len(models)).detach().cpu().numpy()
                    preds_list.append((probs_mean >= 0.5).astype(int))
                    trues_list.append(yb.numpy())

                ender.record()
                torch.cuda.synchronize()
                total_ms = starter.elapsed_time(ender)  # ***ms***
            else:
                t0 = time.perf_counter()
                for xb, yb in test_loader:
                    xb = xb.to(device)
                    probs_accum = None
                    for m in models:
                        logits = m(xb).view(-1)
                        probs = torch.sigmoid(logits)
                        probs_accum = probs if probs_accum is None else probs_accum + probs
                    probs_mean = (probs_accum / len(models)).detach().cpu().numpy()
                    preds_list.append((probs_mean >= 0.5).astype(int))
                    trues_list.append(yb.numpy())
                t1 = time.perf_counter()
                total_ms = (t1 - t0) * 1000.0  # ***ms***

            y_pred = np.concatenate(preds_list).astype(int)
            y_true = np.concatenate(trues_list).astype(int)

            N = max(1, len(y_true))
            K = len(models)

            per_sample_ms = total_ms / N  # 앙상블 기준 ms/sample
            per_model_ms = total_ms / (N * K)  # 모델 1개 환산 ms/sample/model

            m = evaluate_metrics(y_true, y_pred)
            m["InferenceTotal(ms)"] = total_ms
            m["InferenceTime(ms_per_sample)"] = per_sample_ms
            m["InferencePerModel(ms_per_sample_per_model)"] = per_model_ms
            all_fold_metrics.append(m)

            print(
                f"PD={m['PD']:.4f} PF={m['PF']:.4f} FIR={m['FIR']:.4f} Bal={m['Balance']:.4f} | "
                f"Ensemble: {m['InferenceTime(ms_per_sample)']:.3f} ms/sample, "
                f"{m['InferenceTotal(ms)']:.3f} ms total | "
                f"Per-model: {m['InferencePerModel(ms_per_sample_per_model)']:.3f} ms/sample/model"
            )

    # ===== Fold 종료 후 집계/저장 =====
    df = pd.DataFrame(all_fold_metrics)
    mean = df.mean(numeric_only=True)

    print("\n=== [최종 평균 성능] ===")
    print(f"PD={mean['PD']:.4f}, PF={mean['PF']:.4f}, FIR={mean['FIR']:.4f}, Balance={mean['Balance']:.4f}")
    print(
        f"Ensemble Inference: {mean['InferenceTime(ms_per_sample)']:.3f} ms/sample, "
        f"{mean['InferenceTotal(ms)']:.3f} ms total | "
        f"Per-model: {mean['InferencePerModel(ms_per_sample_per_model)']:.3f} ms/sample/model"
    )

    mean_row = mean.to_frame().T
    mean_row.index = ["Average"]
    df_with_avg = pd.concat([df, mean_row], axis=0)

    save_dir = os.path.join("results_sdp", "KAN_DEEP")
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{dataset_name}_metrics.csv")
    df_with_avg.to_csv(out_path, index=True)
    print(f" 결과 저장 완료: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_files = [sys.argv[1]]
    else:
        root_data_dir = "data"
        csv_files = get_all_csv_paths(root_data_dir)

    print(f"총 {len(csv_files)}개의 CSV 파일을 발견했습니다.")
    for csv_file in csv_files:
        print(f"\n===== {csv_file} 데이터셋 실험 시작 =====")
        try:
            run_deep_ensemble_final(csv_file)
        except Exception as e:
            print(f" {csv_file} 처리 중 오류 발생: {e}")
