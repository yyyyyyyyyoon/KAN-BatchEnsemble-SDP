import os
import json
import torch
import torch.nn as nn
from preprocess import preprocess_data

# MLP 정의
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

# 파라미터 수 계산 함수
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
json_dir = os.path.join(base_dir, "..", "mlp_results")
csv_root = os.path.join(base_dir, "..", "data")
json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

param_counts = []

for json_file in json_files:
    dataset_name = os.path.splitext(json_file)[0]
    json_path = os.path.join(json_dir, json_file)

    with open(json_path, "r") as f:
        data = json.load(f)

    params = data["params"]

    # CSV 경로 탐색
    found_csv = None
    for root, _, files in os.walk(csv_root):
        for file in files:
            if file.endswith(".csv") and dataset_name.lower() in file.lower():
                found_csv = os.path.join(root, file)
                break
        if found_csv:
            break

    if not found_csv:
        print(f"❌ {dataset_name} 에 대한 CSV 파일을 찾지 못함")
        continue

    # input_dim 확인
    splits = preprocess_data(found_csv)
    if dataset_name not in splits:
        print(f"❌ 전처리 결과에 {dataset_name} 키가 없음")
        continue

    X_train = splits[dataset_name]["X_train"]
    input_dim = len(X_train[0])

    # MLP 모델 생성 및 파라미터 계산
    model = MLP(input_dim, params["hidden1"], params["hidden2"])
    param_count = count_parameters(model)
    param_counts.append(param_count)
    print(f"{dataset_name}: {param_count:,} parameters")

# 평균 출력
if param_counts:
    avg = sum(param_counts) / len(param_counts)
    print(f"\n📊 평균 파라미터 수: {avg:,.0f}")
