import json
import os
import sys

from optuna.samplers import TPESampler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import time
from pathlib import Path
from math import sqrt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

from KAN_BE.KAN_BatchEnsemble_model import build_kan_ensemble_model

# -------------------- utils --------------------
def print_gpu_info():
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU is not available. Using CPU.")

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_xy(csv_path: str | os.PathLike, target_col: str = "class"):
    p = Path(csv_path)
    df = pd.read_csv(p)
    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' 컬럼을 찾을 수 없습니다.")
    X = df.drop(columns=[target_col]).to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy().astype(np.int64)
    return X, y

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

def make_fold_loaders(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int,
    seed: int = 42,
    use_pin_memory: bool | None = None,
):
    if use_pin_memory is None:
        use_pin_memory = torch.cuda.is_available()

    # split
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    # scale (train fit → test transform)
    scaler = MinMaxScaler()
    X_tr_scaled = scaler.fit_transform(X_tr).astype(np.float32)
    X_te_scaled = scaler.transform(X_te).astype(np.float32)

    # SMOTE (train만)
    smote = SMOTE(random_state=seed)
    try:
        X_tr_res, y_tr_res = smote.fit_resample(X_tr_scaled, y_tr)
    except ValueError:
        X_tr_res, y_tr_res = X_tr_scaled, y_tr

    # tensor
    X_tr_t = torch.from_numpy(X_tr_res)
    y_tr_t = torch.from_numpy(y_tr_res).float()  # BCEWithLogitsLoss → float(0/1)
    X_te_t = torch.from_numpy(X_te_scaled)
    y_te_t = torch.from_numpy(y_te).float()

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        TensorDataset(X_te_t, y_te_t),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_pin_memory,
    )
    return train_loader, test_loader, X_tr.shape[1]

# -------------------- objective --------------------
def objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray, device: torch.device, seed: int = 42):
    # 하이퍼파라미터 샘플링
    d_block   = trial.suggest_categorical("d_block", [64, 128, 256])
    n_blocks  = trial.suggest_int("n_blocks", 1, 5)
    degree    = trial.suggest_int("degree", 2, 5)
    k         = trial.suggest_categorical("k", [2, 4, 8])
    batch_sz  = trial.suggest_categorical("batch_size", [64, 128, 256])
    lr        = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    epochs    = 50

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    pin = torch.cuda.is_available()

    fold_metrics = []
    fold_infer_totals_ms = []  # 총 추론 시간(ms)만 기록

    for train_idx, test_idx in skf.split(X, y):
        train_loader, test_loader, input_dim = make_fold_loaders(
            X, y, train_idx, test_idx, batch_size=batch_sz, seed=seed, use_pin_memory=pin
        )

        model = build_kan_ensemble_model(
            input_dim=input_dim,
            output_dim=1,  # binary
            k=k, d_block=d_block, n_blocks=n_blocks, degree=degree
        ).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # -------- train --------
        model.train()
        for _ in range(epochs):
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=pin)
                yb = yb.to(device, non_blocking=pin)
                optimizer.zero_grad()
                logits = model(xb).view(-1)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        # -------- eval + latency(ms) --------
        model.eval()
        preds, trues = [], []
        with torch.inference_mode():
            # warm-up (타이머 밖)
            for xb, _ in test_loader:
                _ = model(xb.to(device, non_blocking=pin))
                break

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                starter = torch.cuda.Event(enable_timing=True)
                ender   = torch.cuda.Event(enable_timing=True)
                starter.record()

                for xb, yb in test_loader:
                    xb = xb.to(device, non_blocking=pin)
                    logits = model(xb).view(-1)
                    probs  = torch.sigmoid(logits).detach().cpu().numpy()
                    preds.append((probs >= 0.5).astype(int))
                    trues.append(yb.cpu().numpy())

                ender.record()
                torch.cuda.synchronize()
                total_ms = starter.elapsed_time(ender)  # **ms**
            else:
                t0 = time.perf_counter()
                for xb, yb in test_loader:
                    xb = xb.to(device)
                    logits = model(xb).view(-1)
                    probs  = torch.sigmoid(logits).detach().cpu().numpy()
                    preds.append((probs >= 0.5).astype(int))
                    trues.append(yb.cpu().numpy())
                total_ms = (time.perf_counter() - t0) * 1000.0  # **ms**

        y_pred = np.concatenate(preds)
        y_true = np.concatenate(trues).astype(int)
        fold_metrics.append(evaluate_metrics(y_true, y_pred))
        fold_infer_totals_ms.append(total_ms)

    # 평균 Balance 반환 + 부가정보 저장
    avg_balance = float(np.mean([m["Balance"] for m in fold_metrics]))
    best_fold = max(fold_metrics, key=lambda m: m["Balance"])
    for k_attr, v in best_fold.items():
        trial.set_user_attr(k_attr, float(v))

    trial.set_user_attr("InferenceTotal(ms)", float(np.mean(fold_infer_totals_ms)))  # ✅ ms만

    return avg_balance

# -------------------- main --------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_gpu_info()

    csv_paths = glob.glob("data/**/*.csv", recursive=True)

    for csv_path in csv_paths:
        dataset_name = Path(csv_path).stem
        print(f"\n[Dataset: {dataset_name}]")

        X, y = load_xy(csv_path)

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
        )
        study.optimize(lambda t: objective(t, X, y, device, seed=42),
                       n_trials=60, timeout=3600, gc_after_trial=True)

        best = study.best_trial
        print(f"\nBest trial for {dataset_name}:")
        print(best.params)

        # ---- 베스트 파라미터로 모델 재구성하여 파라미터 수 계산 ----
        bp = best.params
        input_dim = X.shape[1]
        best_model = build_kan_ensemble_model(
            input_dim=input_dim,
            output_dim=1,
            k=bp["k"], d_block=bp["d_block"], n_blocks=bp["n_blocks"], degree=bp["degree"]
        )
        model_params = int(count_parameters(best_model))

        # 결과 저장
        os.makedirs("KAN_BE", exist_ok=True)
        out = {
            "dataset": dataset_name,
            "adjusted_score": float(best.value),      # 평균 Balance
            "PD": best.user_attrs.get("PD"),
            "PF": best.user_attrs.get("PF"),
            "FIR": best.user_attrs.get("FIR"),
            "Balance": best.user_attrs.get("Balance"),
            "params": best.params,                    # 베스트 하이퍼파라미터
            "Model_Params": model_params,             # ✅ 파라미터 수
            "InferenceTotal(ms)": best.user_attrs.get("InferenceTotal(ms)"),  # ✅ ms만
        }
        with open(f"KAN_BE/{dataset_name}.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        print(f" 결과 저장 완료: KAN_BE/{dataset_name}.json | "
              f"Params={model_params:,} | InferTotal(avg)={out['InferenceTotal(ms)']:.1f} ms")
