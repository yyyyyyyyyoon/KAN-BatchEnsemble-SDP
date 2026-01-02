from collections import Counter

from preprocess import preprocess_data

# 전처리 함수 호출
splits = preprocess_data("data/AUDI/ProjectA.csv")

# 'ProjectA' 데이터에 대한 오버샘플링된 결과 가져오기
y_resampled = splits["ProjectA"]["y_train"]

# 클래스 비율 확인
print("✅ 오버샘플링 후 클래스 분포:", Counter(y_resampled))

