# 파일명: experiment_sdp/MLP_BE.py
# 실행:
#   전체: python experiment_sdp/MLP_BE.py
#   단일: python experiment_sdp/MLP_BE.py data/AEEEM/EQ.csv

import os
import sys
import time
import glob
from typing import List

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from mlp import MLP
from KAN_BE.KAN_BatchEnsemble_model import LinearEfficientEnsemble, make_efficient_ensemble

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA 사용 가능:", torch.cuda.is_available())
print("GPU 이름:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

DEFAULTS = {
    "epochs": 50,
    "n_splits": 10,
    "seed": 42,
    "hidden1": 128,
    "hidden2": 64,
    "k": 4,
    "batch_size": 128,
    "lr": 1e-3,
}

def load_xy(csv_path: str, target_col: str = "class"):
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' 컬럼을 찾을 수 없습니다. ({csv_path})")
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

def get_all_csv_paths(root_dir: str):
    return sorted(glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True))

# ===== BatchEnsemble Wrapper =====
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

def build_mlp_ensemble_model(
    input_dim: int,
    output_dim: int,
    k: int,
    hidden1: int,
    hidden2: int,
) -> nn.Module:
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

def run_experiment_for_dataset(
    csv_path: str,
    *,
    epochs=DEFAULTS["epochs"],
    n_splits=DEFAULTS["n_splits"],
    seed=DEFAULTS["seed"],
    hidden1=DEFAULTS["hidden1"],
    hidden2=DEFAULTS["hidden2"],
    k=DEFAULTS["k"],
    batch_size=DEFAULTS["batch_size"],
    lr=DEFAULTS["lr"],
):
    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
    print(f"\n===== [{dataset_name}] (MLP-BatchEnsemble) =====")

    X_all, y_all = load_xy(csv_path)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    all_metrics = []
    torch.backends.cudnn.benchmark = True

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all), start=1):
        print(f"\n[Fold {fold}/{n_splits}]")

        X_tr, X_te = X_all[train_idx], X_all[test_idx]
        y_tr, y_te = y_all[train_idx], y_all[test_idx]

        scaler = MinMaxScaler()
        X_tr_scaled = scaler.fit_transform(X_tr).astype(np.float32)
        X_te_scaled = scaler.transform(X_te).astype(np.float32)

        smote = SMOTE(random_state=seed)
        try:
            X_tr_res, y_tr_res = smote.fit_resample(X_tr_scaled, y_tr)
        except ValueError as e:
            print(f"[WARN] SMOTE 실패: {e} -> 원본 train 사용")
            X_tr_res, y_tr_res = X_tr_scaled, y_tr

        X_tr_t = torch.from_numpy(X_tr_res).float()
        y_tr_t = torch.from_numpy(y_tr_res.astype(np.float32))
        X_te_t = torch.from_numpy(X_te_scaled).float()
        y_te_t = torch.from_numpy(y_te.astype(np.float32))

        train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=batch_size, shuffle=False)

        model = build_mlp_ensemble_model(
            input_dim=X_tr.shape[1],
            output_dim=1,
            k=k,
            hidden1=hidden1,
            hidden2=hidden2,
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best = {"Balance": -1}

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb).view(-1)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            model.eval()
            preds, trues = [], []
            with torch.inference_mode():
                for xb, _ in test_loader:
                    xb = xb.to(device, non_blocking=True)
                    _ = model(xb)
                    break

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    starter = torch.cuda.Event(enable_timing=True)
                    ender = torch.cuda.Event(enable_timing=True)
                    starter.record()

                    for xb, yb in test_loader:
                        xb = xb.to(device, non_blocking=True)
                        logits = model(xb).view(-1)
                        probs = torch.sigmoid(logits).detach().cpu().numpy()
                        preds.append((probs >= 0.5).astype(int))
                        trues.append(yb.cpu().numpy())

                    ender.record()
                    torch.cuda.synchronize()
                    total_ms = starter.elapsed_time(ender)
                else:
                    t0 = time.perf_counter()
                    for xb, yb in test_loader:
                        xb = xb.to(device)
                        logits = model(xb).view(-1)
                        probs = torch.sigmoid(logits).detach().cpu().numpy()
                        preds.append((probs >= 0.5).astype(int))
                        trues.append(yb.cpu().numpy())
                    t1 = time.perf_counter()
                    total_ms = (t1 - t0) * 1000.0

            y_pred = np.concatenate(preds).astype(int)
            y_true = np.concatenate(trues).astype(int)

            m = evaluate_metrics(y_true, y_pred)
            N = max(1, len(y_true))
            m["InferenceTotal(ms)"] = total_ms
            m["InferenceTime(ms_per_sample)"] = total_ms / N

            if epoch % max(1, epochs // 5) == 0 or epoch in (1, epochs):
                print(
                    f"[Epoch {epoch:03d}] loss={epoch_loss/len(train_loader):.4f} | "
                    f"PD={m['PD']:.4f} PF={m['PF']:.4f} FIR={m['FIR']:.4f} Bal={m['Balance']:.4f} | "
                    f"Infer={m['InferenceTime(ms_per_sample)']:.3f} ms/sample "
                    f"({m['InferenceTotal(ms)']:.1f} ms total)"
                )

            if m["Balance"] > best["Balance"]:
                best = m

        all_metrics.append(best)

    df = pd.DataFrame(all_metrics)
    mean = df.mean(numeric_only=True)

    print("\n=== [최종 평균 성능] ===")
    print(f"PD={mean['PD']:.4f} PF={mean['PF']:.4f} FIR={mean['FIR']:.4f} Bal={mean['Balance']:.4f}")
    print(
        f"Inference: {mean['InferenceTime(ms_per_sample)']:.3f} ms/sample | "
        f"{mean['InferenceTotal(ms)']:.1f} ms total"
    )

    save_dir = os.path.join("results_sdp", "MLP_BE")
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{dataset_name}_metrics.csv")
    df_with_avg = pd.concat([df, pd.DataFrame([mean], index=["Average"])])
    df_with_avg.to_csv(out_path, index=True)
    print(f"결과 저장 완료: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower().endswith(".csv"):
        csv_files = [sys.argv[1]]
    else:
        root_data_dir = "data"
        csv_files = get_all_csv_paths(root_data_dir)

    print(f"총 {len(csv_files)}개의 CSV 파일을 발견했습니다.")
    for csv_path in csv_files:
        print(f"\n===== {csv_path} =====")
        try:
            run_experiment_for_dataset(csv_path=csv_path)
        except Exception as e:
            print(f"Error on {csv_path}: {e}")
