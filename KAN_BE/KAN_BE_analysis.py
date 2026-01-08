import os
import sys
import glob
import argparse
from typing import Union, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    log_loss
)
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ===== 프로젝트 루트 추가 =====
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from KAN_BE.KAN_BatchEnsemble_model import build_kan_ensemble_model

# =========================
# 기본 설정
# =========================
DEFAULTS = {
    "epochs": 50,
    "n_splits": 10,
    "seed": 42,
    "d_block": 64,
    "n_blocks": 3,
    "degree": 3,
    "k": 4,
    "batch_size": 128,
    "lr": 1e-3,
    "top_pcts": [5.0, 10.0, 20.0],
    "mu_bins": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 유틸 및 분석 함수
# =========================

def plot_ambiguous_zone_distribution(mu, sigma, y, lo=0.4, hi=0.6):
    """
    μ 구간 [lo, hi] 내에서 실제 결함(1)과 정상(0)의 sigma 분포를 비교 시각화
    """
    mask = (mu >= lo) & (mu <= hi)

    if not np.any(mask):
        print(f"\n[알림] μ 구간 [{lo}, {hi}]에 데이터가 없어 시각화를 건너뜁니다.")
        return

    sigma_zone = sigma[mask]
    y_zone = y[mask]

    df_zone = pd.DataFrame({'sigma': sigma_zone, 'class': y_zone})
    df_zone['label'] = df_zone['class'].map({1: 'Defect (1)', 0: 'Normal (0)'})

    plt.figure(figsize=(12, 5))

    # 1. Boxplot (분포의 사분위수 비교)
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df_zone, x='label', y='sigma', palette='Set2')
    plt.title(f'Sigma Distribution in Mu [{lo}, {hi}]')

    # 2. KDE Plot (밀도 비교) - 라벨이 2종류일 때만 수행
    plt.subplot(1, 2, 2)
    if len(np.unique(y_zone)) >= 2:
        sns.kdeplot(data=df_zone, x='sigma', hue='label', fill=True, palette='Set2')
        plt.title(f'Sigma Density in Mu [{lo}, {hi}]')
    else:
        plt.text(0.5, 0.5, 'Only one class present\nKDE plot skipped',
                 ha='center', va='center')

    plt.tight_layout()
    plt.show()

    # 통계치 출력
    stats = df_zone.groupby('label')['sigma'].describe()
    print(f"\n[Sigma Stats in Mu {lo}~{hi}]")
    print(stats)


def load_xy(csv_path: Union[str, os.PathLike], target_col: str = "class"):
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' 컬럼을 찾을 수 없습니다. ({csv_path})")
    X = df.drop(columns=[target_col]).to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy().astype(np.int64)
    return X, y


def get_all_csv_paths(root_dir: str):
    return sorted(glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True))


@torch.inference_mode()
def predict_with_uncertainty(model: nn.Module, xb: torch.Tensor, k: int):
    base = getattr(model, "model", None)
    if base is None:
        raise RuntimeError("wrapper 구조가 예상과 다릅니다.")

    x = xb.unsqueeze(1).expand(-1, k, -1)
    B, K, D = x.shape
    x_flat = x.reshape(B * K, D)

    logits_flat = base(x_flat)
    logits = logits_flat.view(B, K, -1)
    probs = torch.sigmoid(logits).squeeze(-1)

    p_mean = probs.mean(dim=1)
    p_std = probs.std(dim=1, unbiased=False)
    return p_mean, p_std


def top_sigma_stats(y: np.ndarray, sigma: np.ndarray, top_pct: float) -> Dict[str, Any]:
    y = y.astype(int).reshape(-1)
    sigma = sigma.reshape(-1)
    N = len(y)
    k_val = max(1, int(np.ceil(N * (top_pct / 100.0))))
    idx = np.argsort(-sigma)[:k_val]
    base = float(y.mean())
    top = float(y[idx].mean())
    lift = float(top / (base + 1e-12))
    flag = np.zeros(N, dtype=int)
    flag[idx] = 1
    cm = confusion_matrix(y, flag, labels=[0, 1])
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    return {
        "top_pct": top_pct, "N": N, "top_k": k_val, "base_defect_rate": base,
        "top_defect_rate": top, "lift": lift, "TP(defect&topSigma)": int(TP),
        "FP(nondef&topSigma)": int(FP), "FN(defect&notTop)": int(FN), "TN(nondef&notTop)": int(TN),
        "sigma_mean": float(np.mean(sigma)), "sigma_top_mean": float(np.mean(sigma[idx])),
    }


def spearman_sigma_y(y: np.ndarray, sigma: np.ndarray) -> float:
    df = pd.DataFrame({"y": y.astype(float), "sigma": sigma.astype(float)})
    return float(df.corr(method="spearman").loc["sigma", "y"])


def auc_sigma_only(y: np.ndarray, sigma: np.ndarray) -> Dict[str, float]:
    y = y.astype(int).reshape(-1)
    sigma = sigma.reshape(-1)
    if len(np.unique(y)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    roc = roc_auc_score(y, sigma)
    pr = average_precision_score(y, sigma)
    return {"roc_auc": float(roc), "pr_auc": float(pr)}


def mu_binned_top_sigma(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                        bins: List[float], top_pct: float) -> List[Dict[str, Any]]:
    y = y.astype(int).reshape(-1)
    mu = mu.reshape(-1)
    sigma = sigma.reshape(-1)
    out = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (mu >= lo) & (mu < hi) if i < len(bins) - 2 else (mu >= lo) & (mu <= hi)
        if mask.sum() < 5:
            out.append({"mu_bin": f"[{lo:.1f},{hi:.1f}]", "count": int(mask.sum()),
                        "base_defect_rate": np.nan, "top_defect_rate": np.nan, "lift": np.nan})
            continue
        res = top_sigma_stats(y[mask], sigma[mask], top_pct=top_pct)
        out.append({"mu_bin": f"[{lo:.1f},{hi:.1f}]", "count": int(mask.sum()),
                    "base_defect_rate": res["base_defect_rate"], "top_defect_rate": res["top_defect_rate"],
                    "lift": res["lift"]})
    return out


def logistic_extra_info_test(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> Dict[str, float]:
    y = y.astype(int).reshape(-1)
    mu_input = mu.reshape(-1, 1)
    sigma_input = sigma.reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return {"pr_auc_mu": np.nan, "pr_auc_mu_sigma": np.nan, "logloss_mu": np.nan, "logloss_mu_sigma": np.nan}
    clf1, clf2 = LogisticRegression(max_iter=2000), LogisticRegression(max_iter=2000)
    clf1.fit(mu_input, y)
    clf2.fit(np.concatenate([mu_input, sigma_input], axis=1), y)
    p1, p2 = clf1.predict_proba(mu_input)[:, 1], clf2.predict_proba(np.concatenate([mu_input, sigma_input], axis=1))[:,
                                                 1]
    pr1, pr2 = average_precision_score(y, p1), average_precision_score(y, p2)
    ll1, ll2 = log_loss(y, p1, labels=[0, 1]), log_loss(y, p2, labels=[0, 1])
    return {"pr_auc_mu": float(pr1), "pr_auc_mu_sigma": float(pr2), "logloss_mu": float(ll1),
            "logloss_mu_sigma": float(ll2), "delta_pr_auc": float(pr2 - pr1), "delta_logloss": float(ll2 - ll1)}


# =========================
# 메인 분석 실행 함수
# =========================
def run_sigma_relation_analysis(csv_path: str, **kwargs):
    set_seed(kwargs["seed"])

    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
    print(f"\n{'=' * 80}\n[Dataset] {dataset_name}\n{'=' * 80}")

    X_all, y_all = load_xy(csv_path)
    skf = StratifiedKFold(n_splits=kwargs['n_splits'], shuffle=True, random_state=kwargs['seed'])
    all_mu, all_sigma, all_y = [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all), start=1):
        X_tr, X_te = X_all[train_idx], X_all[test_idx]
        y_tr, y_te = y_all[train_idx], y_all[test_idx]

        scaler = MinMaxScaler()
        X_tr_scaled = scaler.fit_transform(X_tr).astype(np.float32)
        X_te_scaled = scaler.transform(X_te).astype(np.float32)

        if kwargs.get('use_smote', True):
            smote = SMOTE(random_state=kwargs['seed'])
            X_tr_res, y_tr_res = smote.fit_resample(X_tr_scaled, y_tr)
        else:
            X_tr_res, y_tr_res = X_tr_scaled, y_tr

        X_tr_t, y_tr_t = torch.from_numpy(X_tr_res).float(), torch.from_numpy(y_tr_res.astype(np.float32))
        X_te_t, y_te_t = torch.from_numpy(X_te_scaled).float(), torch.from_numpy(y_te.astype(np.float32))

        train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=kwargs['batch_size'], shuffle=True)
        test_loader = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=kwargs['batch_size'], shuffle=False)

        model = build_kan_ensemble_model(input_dim=X_tr.shape[1], output_dim=1, k=kwargs['k'],
                                         d_block=kwargs['d_block'], n_blocks=kwargs['n_blocks'],
                                         degree=kwargs['degree']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=kwargs['lr'])
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        for epoch in range(1, kwargs['epochs'] + 1):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb).view(-1), yb)
                loss.backward()
                optimizer.step()

        model.eval()
        m_list, s_list, y_l = [], [], []
        with torch.inference_mode():
            for xb, yb in test_loader:
                mu, sig = predict_with_uncertainty(model, xb.to(device), k=kwargs['k'])
                m_list.append(mu.cpu().numpy());
                s_list.append(sig.cpu().numpy());
                y_l.append(yb.numpy())

        all_mu.append(np.concatenate(m_list));
        all_sigma.append(np.concatenate(s_list));
        all_y.append(np.concatenate(y_l))
        print(f"Fold {fold} Completed.")

    # --- 집계 분석 ---
    mu_all = np.concatenate(all_mu);
    sig_all = np.concatenate(all_sigma);
    y_all_test = np.concatenate(all_y).astype(int)

    print("\n" + "-" * 80 + "\n[ALL-FOLDS AGGREGATED STATS]\n" + "-" * 80)
    print(f"Base Defect Rate: {y_all_test.mean():.4f}")

    # 상위 Sigma 통계
    for pct in kwargs['top_pcts']:
        res = top_sigma_stats(y_all_test, sig_all, pct)
        print(f"Top σ {pct:>2}%: rate={res['top_defect_rate']:.4f}, lift={res['lift']:.2f}")

    # μ 구간별 Sigma 분석
    binned = mu_binned_top_sigma(y_all_test, mu_all, sig_all, bins=kwargs['mu_bins'], top_pct=10.0)
    for row in binned:
        print(
            f"μ{row['mu_bin']} n={row['count']:5d} | base={row['base_defect_rate']:.4f} top={row['top_defect_rate']:.4f} lift={row['lift']:.2f}")

    # 시각화 실행 (가장 중요한 부분)
    plot_ambiguous_zone_distribution(mu_all, sig_all, y_all_test, lo=0.4, hi=0.6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="data")
    args = parser.parse_args()

    csv_files = [args.csv] if args.csv else get_all_csv_paths(args.data_root)
    for path in csv_files:
        run_sigma_relation_analysis(path, **DEFAULTS)


if __name__ == "__main__":
    main()