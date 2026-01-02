import os
import sys
from datetime import time

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

# ===== 기본 하이퍼파라미터 =====
DEFAULTS = {
    "epochs": 50,
    "n_splits": 10,
    "seed": 42,
    "hidden1": 128,
    "hidden2": 64,
    "lr": 1e-3,
    "batch_size": 128,   # train/test 둘 다 동일 배치 사용
}

# ===== 모델 정의 =====
class MLP(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x)

# ===== 유틸: 파라미터 수 =====
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ===== 평가 지표 (KAN 코드와 동일) =====
def evaluate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TP = cm[1, 1] if cm.shape == (2, 2) else 0
    FN = cm[1, 0] if cm.shape == (2, 2) else 0
    FP = cm[0, 1] if cm.shape == (2, 2) else 0
    TN = cm[0, 0] if cm.shape == (2, 2) else 0

    PD = TP / (TP + FN + 1e-6)                       # Recall
    PF = FP / (FP + TN + 1e-6)                       # False Alarm Rate
    FIR = TP / (TP + FP + 1e-6)                      # Precision
    Balance = 1 - (np.sqrt((1 - PD) ** 2 + PF ** 2) / np.sqrt(2))
    return {"PD": PD, "PF": PF, "FIR": FIR, "Balance": Balance}

# ===== 유틸 =====
def get_all_csv_paths(root_dir):
    csv_paths = []
    for foldername, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".csv"):
                csv_paths.append(os.path.join(foldername, filename))
    return csv_paths

def load_xy(csv_path):
    df = pd.read_csv(csv_path)
    if "class" not in df.columns:
        raise ValueError(f"{csv_path}에 'class' 컬럼이 없습니다.")
    y_all = df["class"].astype(int).values
    X_all = df.drop(columns=["class"]).values.astype(np.float32)
    return X_all, y_all

# ===== 실험 실행 =====
def run_final_mlp_experiment(
    csv_path,
    epochs=DEFAULTS["epochs"],
    n_splits=DEFAULTS["n_splits"],
    seed=DEFAULTS["seed"]
):
    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
    X_all, y_all = load_xy(csv_path)
    input_dim = X_all.shape[1]

    hidden1 = DEFAULTS["hidden1"]
    hidden2 = DEFAULTS["hidden2"]
    lr = DEFAULTS["lr"]
    batch_sz = DEFAULTS["batch_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA 사용 가능:", torch.cuda.is_available())
    print("GPU 이름:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all), start=1):
        print(f"\n[Fold {fold}/{n_splits}]")
        X_tr, X_te = X_all[train_idx], X_all[test_idx]
        y_tr, y_te = y_all[train_idx], y_all[test_idx]

        # 1) Scale (train fit → test transform)
        scaler = MinMaxScaler()
        X_tr_scaled = scaler.fit_transform(X_tr).astype(np.float32)
        X_te_scaled = scaler.transform(X_te).astype(np.float32)

        # 2) SMOTE (train only)
        smote = SMOTE(random_state=seed)
        X_tr_res, y_tr_res = smote.fit_resample(X_tr_scaled, y_tr)

        # Tensor/DataLoader
        X_tr_t = torch.from_numpy(X_tr_res).float()
        y_tr_t = torch.from_numpy(y_tr_res).float()
        X_te_t = torch.from_numpy(X_te_scaled).float()
        y_te_t = torch.from_numpy(y_te).float()

        train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_sz, shuffle=True)
        test_loader  = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=batch_sz, shuffle=False)

        # Model/Opt/Loss
        model = MLP(input_dim, hidden1, hidden2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        n_params = count_parameters(model)

        # Train
        model.train()
        for epoch in range(epochs):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb).view(-1)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        # Eval (no time measurement)
        model.eval()
        preds, trues = [], []
        with torch.inference_mode():
            # warm-up (타이머 밖)
            for xb, _ in test_loader:
                _ = model(xb.to(device, non_blocking=True))
                break

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                starter = torch.cuda.Event(enable_timing=True)
                ender   = torch.cuda.Event(enable_timing=True)
                starter.record()

                for xb, yb in test_loader:
                    xb = xb.to(device, non_blocking=True)
                    logits = model(xb).view(-1)
                    probs = torch.sigmoid(logits).detach().cpu().numpy()
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
                    probs = torch.sigmoid(logits).detach().cpu().numpy()
                    preds.append((probs >= 0.5).astype(int))
                    trues.append(yb.cpu().numpy())
                t1 = time.perf_counter()
                total_ms = (t1 - t0) * 1000.0  # **ms**

        y_pred = np.concatenate(preds).astype(int)
        y_true = np.concatenate(trues).astype(int)

        metrics = evaluate_metrics(y_true, y_pred)
        N = max(1, len(y_true))
        metrics["InferenceTotal(ms)"] = total_ms
        metrics["InferenceTime(ms_per_sample)"] = total_ms / N
        metrics["Params"] = n_params

        all_fold_metrics.append(metrics)

        print(
            f"PD={metrics['PD']:.4f} PF={metrics['PF']:.4f} "
            f"FIR={metrics['FIR']:.4f} Bal={metrics['Balance']:.4f} | "
            f"Infer={metrics['InferenceTime(ms_per_sample)']:.3f} ms/sample "
            f"({metrics['InferenceTotal(ms)']:.1f} ms total) | "
            f"Params={n_params:,}"
        )

    # 결과 평균
    df = pd.DataFrame(all_fold_metrics)
    mean_metrics = df.mean(numeric_only=True)

    print("\n=== [최종 평균 성능] ===")
    print(f"PD: {mean_metrics['PD']:.4f}, PF: {mean_metrics['PF']:.4f}, "
          f"FIR: {mean_metrics['FIR']:.4f}, Balance: {mean_metrics['Balance']:.4f}")
    print(f"Inference: {mean_metrics['InferenceTime(ms_per_sample)']:.3f} ms/sample | "
          f"{mean_metrics['InferenceTotal(ms)']:.1f} ms total | "
          f"Params≈{int(mean_metrics['Params']):,}")

    df_with_avg = pd.concat([df, pd.DataFrame([mean_metrics], index=["Average"])])

    # 결과 저장
    os.makedirs("MLP", exist_ok=True)
    result_path = f"MLP/{dataset_name}_metrics.csv"
    df_with_avg.to_csv(result_path, index=True)

    print(f" 결과 저장 완료: {result_path}")

# ===== Entry Point =====
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 명령줄 인자 제공 시 → 단일 파일 실행
        csv_files = [sys.argv[1]]
    else:
        # 인자가 없으면 전체 실행
        root_data_dir = "data"
        csv_files = get_all_csv_paths(root_data_dir)

    print(f"총 {len(csv_files)}개의 CSV 파일을 발견했습니다.")
    for csv_file in csv_files:
        print(f"\n===== {csv_file} 데이터셋 실험 시작 =====")
        try:
            run_final_mlp_experiment(csv_file)
        except Exception as e:
            print(f" {csv_file} 처리 중 오류 발생: {e}")
