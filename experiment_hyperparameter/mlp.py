import glob
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import json
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from preprocess import preprocess_data
import optuna

# MLP 모델 정의
class MLP(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )

    def forward(self, x):
        return self.net(x)

# 평가 지표
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

# 목적 함수
def objective(trial, X_all, y_all, input_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden1 = trial.suggest_int("hidden1", 32, 128, step=32)
    hidden2 = trial.suggest_int("hidden2", 16, 64, step=16)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    metrics_list = []

    for train_idx, test_idx in kf.split(X_all):
        X_tr, X_te = X_all[train_idx], X_all[test_idx]
        y_tr, y_te = y_all[train_idx], y_all[test_idx]

        train_loader = DataLoader(TensorDataset(torch.tensor(X_tr).float(), torch.tensor(y_tr).float()), batch_size=batch_size, shuffle=True)
        X_te_tensor = torch.tensor(X_te).float().to(device)
        y_te_tensor = torch.tensor(y_te).float()

        model = MLP(input_dim=input_dim, hidden1=hidden1, hidden2=hidden2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        for _ in range(10):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                output = model(xb).view(-1)
                loss = criterion(output, yb)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X_te_tensor).view(-1)).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            metrics_list.append(evaluate_metrics(y_te_tensor, preds))

    avg_balance = np.mean([m["Balance"] for m in metrics_list])
    final_metrics = metrics_list[np.argmax([m["Balance"] for m in metrics_list])]
    for key, value in final_metrics.items():
        trial.set_user_attr(key, value)

    return avg_balance

# 메인 실행
if __name__ == "__main__":
    dataset_paths = glob.glob("data/**/*.csv", recursive=True)
    splits = preprocess_data(dataset_paths)

    for dataset_name, data in splits.items():
        X_all = np.array(data["X_train"])
        y_all = np.array(data["y_train"], dtype=int)
        input_dim = X_all.shape[1]

        print(f"\n[Dataset: {dataset_name}]")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X_all, y_all, input_dim), n_trials=100)

        best = study.best_trial
        result_path = f"mlp_results/{dataset_name}.json"
        os.makedirs("mlp_results", exist_ok=True)

        with open(result_path, "w") as f:
            json.dump({
                "dataset": dataset_name,
                "score": best.value,
                "PD": best.user_attrs.get("PD"),
                "PF": best.user_attrs.get("PF"),
                "FIR": best.user_attrs.get("FIR"),
                "Balance": best.user_attrs.get("Balance"),
                "params": best.params
            }, f, indent=2)
        print(f" 결과 저장 완료: {result_path}")
