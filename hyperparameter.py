import json
import os
import glob
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
import pandas as pd

from preprocess import preprocess_data
from model import build_kan_ensemble_model
from sklearn.metrics import confusion_matrix

# ========================= 평가 지표 계산 =========================
def evaluate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TP, FP, FN, TN = cm[1, 1], cm[0, 1], cm[1, 0], cm[0, 0]

    FI = (TP + FP) / (TP + FP + TN + FN)
    PD = TP / (TP + FN) if (TP + FN) > 0 else 0
    PF = FP / (FP + TN) if (FP + TN) > 0 else 0
    FIR = (PD - FI) / PD if PD > 0 else 0
    Balance = 1 - (sqrt((0 - PF) ** 2 + (1 - PD) ** 2) / sqrt(2))

    return {
        "PD": PD,
        "PF": PF,
        "FIR": FIR,
        "Balance": Balance
    }

# ========================= 목적 함수 =========================
def objective(trial, X, y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 하이퍼파라미터 샘플링
    d_block = trial.suggest_categorical("d_block", [64, 128, 256])
    n_blocks = trial.suggest_int("n_blocks", 1, 5)
    degree = trial.suggest_int("degree", 2, 5)
    k = trial.suggest_categorical("k", [2, 4, 8])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    model = build_kan_ensemble_model(
        input_dim=X.shape[1], output_dim=1, k=k,
        d_block=d_block, n_blocks=n_blocks, degree=degree
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y.to_numpy()).float()
    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(50):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb).squeeze()
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_tensor.to(device)).squeeze()
        preds = (torch.sigmoid(logits).cpu().numpy() >= 0.5).astype(int)
        metrics = evaluate_metrics(y, preds)

    # 최적화 기준: Balance
    for k, v in metrics.items():
        trial.set_user_attr(k, v)

    return metrics["Balance"]


# ========================= 최적화 루프 =========================
if __name__ == "__main__":
    dataset_paths = glob.glob("data/**/*.csv", recursive=True)
    splits = preprocess_data(dataset_paths)

    os.makedirs("results", exist_ok=True)

    for name, data in splits.items():
        print(f"\n[Dataset: {name}]")

        X, y = data["X_train"], data["y_train"]

        study = optuna.create_study(direction="maximize")


        best = study.best_trial
        print(f"\nBest trial for {name}:")
        print(best.params)

        # 결과 저장
        result_path = f"results/{name}.json"
        with open(result_path, "w") as f:
            json.dump({
                "dataset": name,
                "adjusted_score": best.value,
                "PD": best.user_attrs.get("PD"),
                "PF": best.user_attrs.get("PF"),
                "FIR": best.user_attrs.get("FIR"),
                "Balance": best.user_attrs.get("Balance"),
                "params": best.params,
            }, f, indent=2)
        print(f" 결과 저장 완료: {result_path}")
