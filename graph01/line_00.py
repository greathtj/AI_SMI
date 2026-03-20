import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 1. 시계열 데이터 생성 (제조 현장 시나리오)
# 10분 간격으로 100개의 온도 센서 데이터를 생성한다.
np.random.seed(42)
data_size = 100
start_time = datetime(2024, 3, 20, 8, 0, 0)
time_list = [start_time + timedelta(minutes=10 * i) for i in range(data_size)]

# 정상 범위: 평균 50도, 표준편차 1인 가우시안 노이즈
sensor_values = 50 + np.random.normal(0, 1, size=data_size)

# 2. 인위적 이상치(Anomaly) 주입
# 특정 시점에 설비 과열 및 냉각 상황을 가정한다.
sensor_values[30] = 56.5  # 과열
sensor_values[75] = 43.2  # 냉각
sensor_values[76] = 44.5  # 연속 이상치

df = pd.DataFrame({'timestamp': time_list, 'temp': sensor_values})

# 3. 시각화 설정 (고급 스타일 적용)
plt.style.use('seaborn-v0_8-whitegrid') # 깔끔한 화이트 그리드 스타일 적용
fig, ax = plt.subplots(figsize=(14, 7), facecolor='#f8fafc')
ax.set_facecolor('#f8fafc')

# (1) 관리 한계선(UCL, LCL) 및 안정 구역 채우기
ucl = 54.0
lcl = 46.0
ax.axhline(y=ucl, color='#e11d48', linestyle='--', linewidth=1.2, alpha=0.8, label='UCL (Upper Control Limit)')
ax.axhline(y=lcl, color='#e11d48', linestyle='--', linewidth=1.2, alpha=0.8, label='LCL (Lower Control Limit)')

# 정상 범위를 연한 푸른색으로 채워 가독성을 높인다.
ax.fill_between(df['timestamp'], lcl, ucl, color='#3b82f6', alpha=0.05, label='Normal Operating Zone')

# (2) 기본 선 그래프: 공정의 전체적인 흐름을 매끄럽게 표현한다.
ax.plot(df['timestamp'], df['temp'], color='#475569', linewidth=2, alpha=0.8, label='Sensor Reading', zorder=3)

# (3) 이상치 추출 및 강조 (조명 비추기)
# 한계를 벗어난 지점에 테두리가 있는 붉은 점을 찍어 시선을 끈다.
outliers = df[(df['temp'] > ucl) | (df['temp'] < lcl)]
ax.scatter(outliers['timestamp'], outliers['temp'], 
           color='#fbbf24', s=120, marker='o', edgecolors='#e11d48', linewidths=2, 
           zorder=5, label='Anomaly Detected')

# 4. 그래프 디테일 다듬기
ax.set_title('Industrial Sensor Monitoring: Temperature Anomaly Detection', fontsize=18, fontweight='bold', pad=25, color='#1e293b')
ax.set_xlabel('Operation Time (Timestamp)', fontsize=12, fontweight='medium', color='#475569')
ax.set_ylabel('Temperature (Celsius)', fontsize=12, fontweight='medium', color='#475569')

# Y축 범위 및 그리드 정돈
ax.set_ylim(40, 60)
ax.grid(True, linestyle='--', alpha=0.4)

# 불필요한 테두리 제거 (Spines)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# 범례 설정: 깔끔한 디자인으로 우측 상단 배치
ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, edgecolor='#e2e8f0', fontsize=10)

# 시간 라벨 정리: 45도 회전 및 폰트 크기 조정
plt.xticks(rotation=45, fontsize=10, color='#64748b')
plt.yticks(fontsize=10, color='#64748b')

plt.tight_layout()
plt.show()
