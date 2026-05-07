import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 생성 단계
np.random.seed(42)
time = np.arange(0, 20, 1)

# 온도가 서서히 상승하는 주조 공정 상황을 가정하여 데이터 생성
# 정수형으로 계산 후 .astype(float)를 사용하여 NaN(결측치)이 들어갈 수 있도록 설정
temp = (1000 + (time * 10) + np.random.randint(-5, 5, size=len(time))).astype(float)

# 센서 오류 등으로 인해 12분~15분 사이의 데이터가 유실되었다고 가정 (결측치 삽입)
temp[12:16] = np.nan 

df = pd.DataFrame({'Time': time, 'Temperature': temp})

# 2. 결측치 처리 방법 적용
# 방법 A: 직전 값 채우기 (Forward Fill) -> 마지막으로 확인된 값을 그대로 복사
df['FFill'] = df['Temperature'].ffill()

# 방법 B: 선형 보간법 (Linear Interpolation) -> 끊긴 두 점 사이를 직선으로 연결
df['Interpolation'] = df['Temperature'].interpolate(method='linear')

# 3. 시각화 단계
plt.figure(figsize=(12, 6))

# [원본] 실제 측정값: 결측치 때문에 중간에 점들이 끊겨서 보임
plt.plot(df['Time'], df['Temperature'], 'ko', label='Actual Measurements (NaN)', markersize=8)

# [FFill] 직전 값 채우기: 계단 모양으로 유지되다가 다음 값이 나오면 점프함
plt.plot(df['Time'], df['FFill'], color='red', linestyle='--', marker='x', label='Forward Fill (FFill)')

# [Interpolation] 선형 보간법: 끊어진 구간을 부드러운 직선으로 연결함
plt.plot(df['Time'], df['Interpolation'], color='blue', linestyle='-', marker='s', label='Linear Interpolation')

# 그래프 레이블 및 제목 설정 (영문)
plt.title('Comparison of Missing Value Imputation: FFill vs Linear Interpolation', fontsize=15)
plt.xlabel('Time (min)', fontsize=12)
plt.ylabel('Temperature (℃)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
