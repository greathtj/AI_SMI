import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 시뮬레이션용 가상 데이터 생성 (센서 데이터)
# 분석 흐름을 유지하며 빠르게 데이터를 확인하는 과정을 재현한다.
np.random.seed(42)
dates = pd.date_range(start='2024-01-01', periods=100)
sensor_data = {
    'Temp_Main': np.random.normal(70, 2, 100),      # 정상 범위 온도
    'Temp_Sub': np.random.normal(65, 5, 100),       # 변동성이 큰 서브 온도
    'Pressure': np.random.uniform(10, 20, 100)      # 일정 범위의 압력값
}

# 데이터프레임 생성
df = pd.DataFrame(sensor_data, index=dates)

# 데이터에 인위적으로 이상치(Outlier) 주입 (초벌 확인용)
df.iloc[50:55, 0] = 95  # Temp_Main에 갑작스러운 고온 발생

print("--- Data Preview ---")
print(df.head())

# 2. Pandas 내장 시각화 예제 (가장 빠른 확인용)

# [A] 선 그래프 (Line Plot): 전체적인 추세와 튀는 값(이상치)을 즉시 확인
# 별도의 시각화 라이브러리 설정 없이 .plot() 한 줄로 해결
df.plot(figsize=(12, 5), title="Sensor Trend Analysis (Line Plot)")
plt.ylabel("Value")
plt.grid(True, linestyle='--', alpha=0.6)

# [B] 히스토그램 (Histogram): 데이터가 특정 구간에 얼마나 몰려있는지(분포) 확인
# bins 옵션 하나로 밀도를 조절하여 데이터의 민낯을 보다.
df[['Temp_Main', 'Temp_Sub']].plot(kind='hist', bins=30, alpha=0.5, figsize=(10, 5), title="Data Distribution (Histogram)")

# [C] 박스 플롯 (Box Plot): 이상치(Outlier)의 통계적 위치를 한눈에 파악
# 데이터의 사분위수를 계산할 필요 없이 즉시 시각화한다.
df.plot(kind='box', figsize=(10, 5), title="Outlier Detection (Box Plot)")

# 모든 그래프 출력
plt.tight_layout()
plt.show()

"""
[실무 활용 팁]
1. df.plot(): 선 그래프 (디폴트) - 시계열 데이터 추세 확인용
2. df.plot(kind='hist'): 히스토그램 - 데이터 편중 및 빈도 확인용
3. df.plot(kind='box'): 박스 플롯 - 이상치 유무 및 통계적 분포 확인용
4. df.plot(kind='bar'): 바 차트 - 카테고리별 비교용
"""
