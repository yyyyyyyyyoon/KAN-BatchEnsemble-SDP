import matplotlib.pyplot as plt
import numpy as np

# 예시 데이터 (실제 값으로 교체)
params_kan = [42357, 42357, 42357, 35889, 35889, 35889, 37737, 37737, 37737]
time_kan = [1.69, 2.04, 1.72, 3.96, 4.21, 6.18, 1.83, 1.6, 1.46]

params_be = [45677, 45677, 45677, 39013, 39013, 39013, 40917, 40917, 40917]
time_be = [2.51, 2.91, 2.38, 3.14, 3.65, 5.24, 1.39, 1.7, 1.85]

params_deep = [211785, 211785, 211785, 179445, 179445, 179445, 188685, 188685, 188685]
time_deep = [5.01, 5.16, 5.65, 9.55, 10.83, 17.4, 6.44, 5.38, 4.29]

plt.figure(figsize=(10, 7))
plt.scatter(params_kan, time_kan, color='blue', label='KAN', alpha=0.7, s=90)
plt.scatter(params_be, time_be, color='green', marker='s', label='KAN-BatchEnsemble', alpha=0.7, s=90)
plt.scatter(params_deep, time_deep, color='red', marker='^', label='DeepEnsemble-KAN', alpha=0.7, s=90)

# 평균값 표시
plt.scatter(np.mean(params_kan), np.mean(time_kan), color='blue', edgecolor='black', s=320, marker='*')
plt.scatter(np.mean(params_be), np.mean(time_be), color='green', edgecolor='black', s=320, marker='*')
plt.scatter(np.mean(params_deep), np.mean(time_deep), color='red', edgecolor='black', s=320, marker='*')

plt.xscale('log')
plt.xlabel('Parameters (log scale)')
plt.ylabel('Inference Time (ms)')
plt.title('모델별 효율성 비교 (파라미터 수 vs 추론 시간)')
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
plt.show()
