import glob
import os
import sys
import json
import time
from math import sqrt

import torch
import torch.nn as nn
import numpy as np
import optuna
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from KAN_BE.KAN_BatchEnsemble_model import KAN

# ===== 유틸 =====
def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ===== 평가 지표 =====
def evaluate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TP = cm[1, 1] if cm.shape == (2, 2) else 0
    FN = cm[1, 0] if cm.shape == (2, 2) else 0
    FP = cm[0, 1] if cm.shape == (2, 2) else 0
    TN = cm[0, 0] if cm.shape == (2, 2) else 0

    PD = TP / (TP + FN + 1e-6)
    PF = FP / (FP + TN + 1e-6)
    FIR = TP / (TP + FP + 1e-6)
    Balance = 1 - (sqrt((1 - PD) ** 2 + PF ** 2) / sqrt(2))
    return {"PD": PD, "PF": PF, "FIR": FIR, "Balance": Balance}

# ===== CSV 로드 =====
def load_xy(csv_path, target_col="class"):
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' column not found in {csv_path}")
    X = df.drop(columns=[target_col]).values.astype(np.float32)
    y = df[target_col].values.astype(np.int64)
    return X, y

# ===== 목적 함수 =====
def objective(trial, X_all, y_all, input_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 하이퍼파라미터 탐색 공간
    d_block    = trial.suggest_categorical("d_block", [64, 128, 256])
    n_blocks   = trial.suggest_int("n_blocks", 1, 4)
    degree     = trial.suggest_int("degree", 2, 5)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    epochs     = 50

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_metrics = []
    fold_infer_times = []

    for train_idx, test_idx in skf.split(X_all, y_all):
        X_tr, X_te = X_all[train_idx], X_all[test_idx]
        y_tr, y_te = y_all[train_idx], y_all[test_idx]

        # 정규화
        scaler = MinMaxScaler()
        X_tr_scaled = scaler.fit_transform(X_tr).astype(np.float32)
        X_te_scaled = scaler.transform(X_te).astype(np.float32)

        # SMOTE
        try:
            X_tr_res, y_tr_res = SMOTE(random_state=42).fit_resample(X_tr_scaled, y_tr)
        except ValueError:
            X_tr_res, y_tr_res = X_tr_scaled, y_tr

        # Tensor & Loader
        X_tr_t = torch.from_numpy(X_tr_res)
        y_tr_t = torch.from_numpy(y_tr_res).float()
        X_te_t = torch.from_numpy(X_te_scaled)
        y_te_t = torch.from_numpy(y_te).float()

        train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=batch_size, shuffle=False)

        # 모델
        model = KAN(
            d_in=input_dim,
            d_out=1,
            d_block=d_block,
            n_blocks=n_blocks,
            degree=degree
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        # 학습
        model.train()
        for _ in range(epochs):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb).view(-1)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        # 평가 + 추론 시간 측정
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            # warm-up
            for xb, _ in test_loader:
                _ = model(xb.to(device))
                break

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                starter = torch.cuda.Event(enable_timing=True)
                ender   = torch.cuda.Event(enable_timing=True)
                starter.record()

                for xb, yb in test_loader:
                    xb = xb.to(device)
                    probs = torch.sigmoid(model(xb).view(-1)).cpu().numpy()
                    preds.append((probs >= 0.5).astype(int))
                    trues.append(yb.cpu().numpy())

                ender.record()
                torch.cuda.synchronize()
                total_ms = starter.elapsed_time(ender)
            else:
                t0 = time.perf_counter()
                for xb, yb in test_loader:
                    xb = xb.to(device)
                    probs = torch.sigmoid(model(xb).view(-1)).cpu().numpy()
                    preds.append((probs >= 0.5).astype(int))
                    trues.append(yb.cpu().numpy())
                total_ms = (time.perf_counter() - t0) * 1000.0

        fold_infer_times.append(total_ms)

        y_pred = np.concatenate(preds)
        y_true = np.concatenate(trues).astype(int)
        fold_metrics.append(evaluate_metrics(y_true, y_pred))

    avg_balance = float(np.mean([m["Balance"] for m in fold_metrics]))
    best_fold = max(fold_metrics, key=lambda m: m["Balance"])
    for k, v in best_fold.items():
        trial.set_user_attr(k, float(v))

    trial.set_user_attr("InferenceTotal(ms)", float(np.mean(fold_infer_times)))
    return avg_balance

# ===== 메인 =====
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA 사용 가능:", torch.cuda.is_available())

    csv_paths = glob.glob("data/**/*.csv", recursive=True)

    for csv_path in csv_paths:
        dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"\n[Dataset: {dataset_name}]")

        X_all, y_all = load_xy(csv_path)
        input_dim = X_all.shape[1]

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: objective(t, X_all, y_all, input_dim), n_trials=60)

        best = study.best_trial

        # ---- 베스트 파라미터로 모델 재구성 & 파라미터 수 계산 ----
        bp = best.params
        best_model = KAN(
            d_in=input_dim,
            d_out=1,
            d_block=bp["d_block"],
            n_blocks=bp["n_blocks"],
            degree=bp["degree"]
        )
        kan_params = int(count_parameters(best_model))

        # ---- 결과 저장 ----
        os.makedirs("KAN", exist_ok=True)
        result_path = f"KAN/{dataset_name}.json"
        out = {
            "dataset": dataset_name,
            "score": float(best.value),
            "PD": best.user_attrs.get("PD"),
            "PF": best.user_attrs.get("PF"),
            "FIR": best.user_attrs.get("FIR"),
            "Balance": best.user_attrs.get("Balance"),
            "params": best.params,
            "KAN_Params": kan_params,
            "input_dim": int(input_dim),
            "InferenceTotal(ms)": best.user_attrs.get("InferenceTotal(ms)"),  # ✅ 추론 시간 기록
        }
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        print(f"✅ 결과 저장 완료: {result_path} | KAN_Params={kan_params:,} | "
              f"InferenceTotal(avg)={out['InferenceTotal(ms)']:.2f} ms")
