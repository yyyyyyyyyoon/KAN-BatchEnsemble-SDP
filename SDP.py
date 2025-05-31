# 실행 명령어 python SDP.py data/AEEEM/EQ.csv

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from preprocess import preprocess_data
from model import build_kan_ensemble_model

# 평가 지표 계산 함수 정의
def evaluate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1] if cm.shape == (2, 2) else 0
    FN = cm[1, 0] if cm.shape == (2, 2) else 0
    FP = cm[0, 1] if cm.shape == (2, 2) else 0
    TN = cm[0, 0] if cm.shape == (2, 2) else 0

    PD = TP / (TP + FN + 1e-6)
    PF = FP / (FP + TN + 1e-6)
    FIR = TP / (TP + FP + 1e-6)
    Balance = 1 - (np.sqrt((1 - PD)**2 + PF**2) / np.sqrt(2))

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return {
        "PD": PD,
        "PF": PF,
        "FIR": FIR,
        "Balance": Balance,
        "Accuracy": acc,
        "F1": f1
    }

# 기본 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
k = 4  # BatchEnsemble 크기
batch_size = 128
epochs = 10

def main():
    # 1. CSV 경로 받아오기
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        print("Error: No CSV file path provided.")
        sys.exit(1)

    # 2. 전처리 수행
    splits = preprocess_data(csv_path)
    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]

    if dataset_name not in splits:
        print(f"Error: Dataset '{dataset_name}' not found in splits.")
        sys.exit(1)

    X_all = np.array(splits[dataset_name]["X_train"])
    y_all = np.array(splits[dataset_name]["y_train"], dtype=int)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_metrics = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_all)):
        print(f"\n[Fold {fold + 1}/10]")

        # Fold별 데이터 나누기
        X_tr, X_te = X_all[train_idx], X_all[test_idx]
        y_tr, y_te = y_all[train_idx], y_all[test_idx]

        # Tensor 변환
        X_tr_tensor = torch.from_numpy(X_tr).float()
        y_tr_tensor = torch.from_numpy(y_tr).float()
        X_te_tensor = torch.from_numpy(X_te).float()
        y_te_tensor = torch.from_numpy(y_te).float()

        # Dataloader
        train_loader = DataLoader(TensorDataset(X_tr_tensor, y_tr_tensor), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_te_tensor, y_te_tensor), batch_size=batch_size)

        # 모델 생성
        model = build_kan_ensemble_model(input_dim=X_tr.shape[1], output_dim=1, k=k).to(device)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # 학습
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                output = model(xb).squeeze()
                loss = criterion(output, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[Epoch {epoch + 1}] Loss: {total_loss / len(train_loader):.4f}")

        # 평가
        model.eval()
        y_pred_all, y_true_all = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb).squeeze()
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                y_pred_all.extend(preds)
                y_true_all.extend(yb.cpu().numpy())

        fold_metrics = evaluate_metrics(y_true_all, y_pred_all)
        print("Fold Metrics:", fold_metrics)
        all_metrics.append(fold_metrics)

    print("\n=== [최종 평균 성능] ===")
    df = pd.DataFrame(all_metrics)
    print(df.mean())

if __name__ == "__main__":
    main()