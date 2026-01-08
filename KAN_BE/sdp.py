import os
import sys
import time
import glob

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

import torch
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# KAN 모듈 경로 (필요 시 수정)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from KAN_BE.KAN_BatchEnsemble_model import build_kan_ensemble_model

# ===== 고정 하이퍼파라미터 =====
DEFAULTS = {
    "epochs": 50,
    "n_splits": 10,
    "seed": 2026,
    "d_block": 64,
    "n_blocks": 3,
    "degree": 3,
    "k": 5,               # BatchEnsemble 내부 전문가 수
    "batch_size": 128,
    "lr": 1e-3,
}

# ===== 디바이스 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA 사용 가능:", torch.cuda.is_available())
print("GPU 이름:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# ===== 유틸 =====
def load_xy(csv_path: str | os.PathLike, target_col: str = "class"):
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
    Balance = 1 - (np.sqrt((1 - PD)**2 + PF**2) / np.sqrt(2))
    return {"PD": PD, "PF": PF, "FIR": FIR, "Balance": Balance}

def get_all_csv_paths(root_dir: str):
    return sorted(glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True))

# ===== 메인 실험 =====
def run_experiment_for_dataset(
    csv_path: str,
    *,
    epochs=DEFAULTS["epochs"],
    n_splits=DEFAULTS["n_splits"],
    seed=DEFAULTS["seed"],
    d_block=DEFAULTS["d_block"],
    n_blocks=DEFAULTS["n_blocks"],
    degree=DEFAULTS["degree"],
    k=DEFAULTS["k"],
    batch_size=DEFAULTS["batch_size"],
    lr=DEFAULTS["lr"],
):
    set_seed(seed)

    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
    print(f"\n===== [{dataset_name}] =====")

    X_all, y_all = load_xy(csv_path)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_metrics = []
    torch.backends.cudnn.benchmark = True

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
        X_tr_res, y_tr_res = smote.fit_resample(X_tr_scaled, y_tr)

        # Tensor & DataLoader
        X_tr_t = torch.from_numpy(X_tr_res).float()
        y_tr_t = torch.from_numpy(y_tr_res).float()
        X_te_t = torch.from_numpy(X_te_scaled).float()
        y_te_t = torch.from_numpy(y_te).float()

        train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=batch_size, shuffle=False)

        # Model (BatchEnsemble 내부 k개)
        model = build_kan_ensemble_model(
            input_dim=X_tr.shape[1],
            output_dim=1,
            k=k,
            d_block=d_block,
            n_blocks=n_blocks,
            degree=degree
        ).to(device)

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best = {"Balance": -1}

        for epoch in range(1, epochs + 1):
            # --- train ---
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

            # --- eval + latency (ms) ---
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
                    ender   = torch.cuda.Event(enable_timing=True)
                    starter.record()

                    for xb, yb in test_loader:
                        xb = xb.to(device, non_blocking=True)
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
                    t1 = time.perf_counter()
                    total_ms = (t1 - t0) * 1000.0         # **ms**

            y_pred = np.concatenate(preds)
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

    # 평균 출력 및 저장
    df = pd.DataFrame(all_metrics)
    mean = df.mean(numeric_only=True)

    print("\n=== [최종 평균 성능] ===")
    print(f"PD={mean['PD']:.4f} PF={mean['PF']:.4f} FIR={mean['FIR']:.4f} Bal={mean['Balance']:.4f}")
    print(f"Inference: {mean['InferenceTime(ms_per_sample)']:.3f} ms/sample | "
          f"{mean['InferenceTotal(ms)']:.1f} ms total")

    save_dir = os.path.join("results_sdp", "KAN_BE")
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{dataset_name}_metrics.csv")

    df_with_avg = pd.concat([df, pd.DataFrame([mean], index=["Average"])])
    df_with_avg.to_csv(out_path, index=True)
    print(f"결과 저장 완료: {out_path}")

# ===== Entry Point =====
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
