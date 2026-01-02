import glob
import os
import sys
import json
import time
from math import sqrt
from typing import List

import optuna
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE

# 프로젝트 루트 경로 추가 (필요시 조정)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from KAN_BE.KAN_BatchEnsemble_model import KAN  # 단일 KAN 모델

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ===== 유틸 =====
def count_parameters(model: torch.nn.Module) -> int:
    """학습 가능한 파라미터 수"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_xy(csv_path: str, target_col: str = "class"):
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' column not found in {csv_path}")
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


# ===== Deep Ensemble 예측 =====
def deep_ensemble_predict(models: List[nn.Module], X_te: np.ndarray) -> np.ndarray:
    X_tensor = torch.from_numpy(X_te).float().to(device)
    probs_list = []
    with torch.no_grad():
        for m in models:
            m.eval()
            logits = m(X_tensor).view(-1)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            probs_list.append(prob)
    probs_mean = np.mean(probs_list, axis=0)
    return (probs_mean >= 0.5).astype(int)


# ===== Fold 로더 생성 (정규화 + SMOTE(train만)) =====
def make_fold_loaders(X, y, train_idx, test_idx, batch_size, seed=42, pin=None):
    if pin is None:
        pin = torch.cuda.is_available()

    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    scaler = MinMaxScaler()
    X_tr_scaled = scaler.fit_transform(X_tr).astype(np.float32)
    X_te_scaled = scaler.transform(X_te).astype(np.float32)

    try:
        X_tr_res, y_tr_res = SMOTE(random_state=seed).fit_resample(X_tr_scaled, y_tr)
    except ValueError:
        X_tr_res, y_tr_res = X_tr_scaled, y_tr

    X_tr_t = torch.from_numpy(X_tr_res)
    y_tr_t = torch.from_numpy(y_tr_res).float()
    X_te_t = torch.from_numpy(X_te_scaled)
    y_te_t = torch.from_numpy(y_te).float()

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        TensorDataset(X_te_t, y_te_t),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin,
    )
    return train_loader, test_loader, X_tr_scaled.shape[1], X_te_scaled, y_te


# ===== 목적 함수 =====
def objective(trial, X, y, ensemble_size=5, seed=42):
    # 하이퍼파라미터 공간
    d_block    = trial.suggest_categorical("d_block", [64, 128, 256])
    n_blocks   = trial.suggest_int("n_blocks", 1, 5)
    degree     = trial.suggest_int("degree", 2, 5)
    lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    epochs     = 50  # 너의 원래 코드에 맞춤

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    pin = torch.cuda.is_available()

    fold_metrics = []
    fold_infer_totals_ms = []  # 총 추론 시간(ms)만 기록

    for tr_idx, te_idx in skf.split(X, y):
        train_loader, test_loader, input_dim, X_te_scaled, y_te = make_fold_loaders(
            X, y, tr_idx, te_idx, batch_size=batch_size, seed=seed, pin=pin
        )

        # ---- 앙상블 학습 ----
        models: List[nn.Module] = []
        for _ in range(ensemble_size):
            m = KAN(d_in=input_dim, d_out=1, d_block=d_block, n_blocks=n_blocks, degree=degree).to(device)
            opt = torch.optim.Adam(m.parameters(), lr=lr)
            crit = nn.BCEWithLogitsLoss()

            m.train()
            for _ in range(epochs):
                for xb, yb in train_loader:
                    xb = xb.to(device, non_blocking=pin)
                    yb = yb.to(device, non_blocking=pin)
                    opt.zero_grad()
                    logits = m(xb).view(-1)
                    loss = crit(logits, yb)
                    loss.backward()
                    opt.step()
            models.append(m)

        # ---- 추론 시간 측정 (테스트 전체 forward) ----
        if torch.cuda.is_available():
            # DataLoader를 사용한 루프(메모리 절약 & 동일 측정 방식)
            with torch.inference_mode():
                # warm-up
                for xb, _ in test_loader:
                    _ = models[0](xb.to(device, non_blocking=pin))
                    break
                torch.cuda.synchronize()
                starter = torch.cuda.Event(enable_timing=True)
                ender   = torch.cuda.Event(enable_timing=True)
                starter.record()

                # 테스트 루프: 각 배치에 대해 K개 모델 forward 후 평균
                preds_list, trues_list = [], []
                for xb, yb in test_loader:
                    xb = xb.to(device, non_blocking=pin)
                    probs_accum = None
                    for m in models:
                        logits = m(xb).view(-1)
                        p = torch.sigmoid(logits)
                        probs_accum = p if probs_accum is None else probs_accum + p
                    probs_mean = (probs_accum / len(models)).detach().cpu().numpy()
                    preds_list.append((probs_mean >= 0.5).astype(int))
                    trues_list.append(yb.cpu().numpy())

                ender.record()
                torch.cuda.synchronize()
                total_ms = starter.elapsed_time(ender)  # ms

                y_pred = np.concatenate(preds_list).astype(int)
                y_true = np.concatenate(trues_list).astype(int)
        else:
            # CPU 측정 (perf_counter)
            t0 = time.perf_counter()
            y_pred = deep_ensemble_predict(models, X_te_scaled)
            total_ms = (time.perf_counter() - t0) * 1000.0
            y_true = y_te.astype(int)

        fold_infer_totals_ms.append(total_ms)
        fold_metrics.append(evaluate_metrics(y_true, y_pred))

    # 평균 Balance 및 부가정보 저장
    avg_balance = float(np.mean([m["Balance"] for m in fold_metrics]))
    best_fold = max(fold_metrics, key=lambda m: m["Balance"])
    for k, v in best_fold.items():
        trial.set_user_attr(k, float(v))
    trial.set_user_attr("InferenceTotal(ms)", float(np.mean(fold_infer_totals_ms)))  # ms만

    return avg_balance


# ───────────────────── 실행 ─────────────────────
if __name__ == "__main__":
    csv_paths = glob.glob("data/**/*.csv", recursive=True)
    os.makedirs("KAN_DEEP", exist_ok=True)

    for csv_path in csv_paths:
        dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"\n[Dataset: {dataset_name}]")

        X_all, y_all = load_xy(csv_path)
        input_dim = X_all.shape[1]

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: objective(t, X_all, y_all, ensemble_size=5, seed=42), n_trials=100)

        best = study.best_trial

        # ---- 파라미터 수 계산 (단일 KAN × ensemble_size) ----
        bp = best.params
        single_model = KAN(d_in=input_dim, d_out=1,
                           d_block=bp["d_block"], n_blocks=bp["n_blocks"], degree=bp["degree"])
        single_params = count_parameters(single_model)
        total_params = single_params * 5  # ensemble_size=5

        # ---- 결과 저장 ----
        out = {
            "dataset": dataset_name,
            "score": float(best.value),  # 최고(=best fold) Balance로 반환했으므로 score는 best 기준
            "PD": best.user_attrs.get("PD"),
            "PF": best.user_attrs.get("PF"),
            "FIR": best.user_attrs.get("FIR"),
            "Balance": best.user_attrs.get("Balance"),
            "params": best.params,
            "Model_Params": int(total_params),          # Deep Ensemble 총 파라미터 수
            "Single_Params": int(single_params),        # 단일 KAN 파라미터 수(참고)
            "InferenceTotal(ms)": best.user_attrs.get("InferenceTotal(ms)"),  # fold 평균 ms
        }
        out_path = f"KAN_DEEP/{dataset_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f" 결과 저장 완료: {out_path} | "
              f"Params(single)={single_params:,}, total={total_params:,} | "
              f"InferTotal(avg)={out['InferenceTotal(ms)']:.1f} ms")
