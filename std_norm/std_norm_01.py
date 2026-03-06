import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 시뮬레이션 데이터 생성
# 평균 100, 표준편차 15인 정규분포 데이터 100개 생성
np.random.seed(42)
raw_data = np.random.normal(loc=100, scale=15, size=100)

# 여러 개의 이상치(Outliers)를 데이터 중간중간에 삽입
# 정규화와 표준화의 차이를 더 극명하게 보여주기 위해 3개의 강한 이상치 추가
outlier_indices = [25, 60, 100] # 이상치가 발생한 시점들
outlier_values = [280, 240, 260] # 이상치 값들

# 리스트 변환 후 특정 위치에 값 삽입
full_data = list(raw_data)
for idx, val in zip(outlier_indices, outlier_values):
    if idx < len(full_data):
        full_data[idx] = val
    else:
        full_data.append(val)

df = pd.DataFrame({
    'Time': np.arange(len(full_data)),
    'Value': full_data
})

# 2. 스케일링 적용
# 정규화 (Min-Max): 이상치가 하나라도 있으면 전체 데이터를 0~0.2 수준으로 압축해버림
min_max_scaler = MinMaxScaler()
df['Normalized'] = min_max_scaler.fit_transform(df[['Value']])

# 표준화 (Standard): 평균과 표준편차를 사용하므로 이상치가 있어도 데이터의 상대적 분포가 유지됨
std_scaler = StandardScaler()
df['Standardized'] = std_scaler.fit_transform(df[['Value']])

# 3. 시각화 (Histogram + Time-series Scatter)
fig, axes = plt.subplots(2, 3, figsize=(18, 10), gridspec_kw={'height_ratios': [1.5, 2]})
fig.suptitle('Scaling Sensitivity Analysis: Multiple Outliers Impact', fontsize=20, fontweight='bold')

colors = ['gray', 'blue', 'green']
columns = ['Value', 'Normalized', 'Standardized']
titles = [
    '1. Raw Data\n(Original Distribution)', 
    '2. Normalization (Min-Max)\n[Extremely Sensitive to Outliers]', 
    '3. Standardization (Z-Score)\n[Robust to Multiple Outliers]'
]
y_labels = ['Value', '0 to 1 Scale', 'Z-Score Scale']

for i in range(3):
    col = columns[i]
    
    # 상단: 히스토그램
    sns.histplot(df[col], kde=True, ax=axes[0, i], color=colors[i])
    axes[0, i].set_title(titles[i], fontsize=13, fontweight='bold', pad=15)
    axes[0, i].set_xlabel('')
    
    # 하단: 시계열 산점도
    axes[1, i].scatter(df['Time'], df[col], alpha=0.5, color=colors[i], s=30)
    
    # 이상치 강조 표시 (빨간색 점)
    actual_outliers = df.iloc[outlier_indices]
    axes[1, i].scatter(actual_outliers['Time'], actual_outliers[col], 
                        color='red', s=100, edgecolors='black', label='Outliers')
    
    axes[1, i].plot(df['Time'], df[col], color=colors[i], alpha=0.15)
    axes[1, i].set_xlabel('Time (Sequence)')
    axes[1, i].set_ylabel(y_labels[i])
    if i == 0:
        axes[1, i].legend()

# 정규화 차트에서 데이터 쏠림(Squashing) 현상 주석 추가
axes[1, 1].annotate('Most data points are forced\ninto a very narrow range (0.0-0.3)', 
                     xy=(50, 0.15), xytext=(10, 0.6),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                     fontsize=11, color='blue', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8))

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 4. 분석 결과 요약
print("--- 다중 이상치 존재 시 스케일링 요약 ---")
summary = df[['Value', 'Normalized', 'Standardized']].describe().loc[['min', 'max', 'mean', '50%']]
print(summary.round(3))

print("\n[핵심 관찰 포인트]")
print("1. 정규화(Min-Max)의 붕괴:")
print(f"   - 최댓값({max(outlier_values)})이 1.0이 되면서, 대부분의 정상 데이터(평균 100 근처)가 {df['Normalized'].mean():.2f} 지점으로 내려앉았다.")
print("   - 그래프를 보면 파란색 점들이 바닥에 거의 붙어 있어 데이터 간의 차이를 구분하기 어렵다.")
print("\n2. 표준화(Z-Score)의 유지력:")
print("   - 이상치가 3개나 섞여 평균과 표준편차에 영향을 주었음에도 불구하고,")
print("   - 정상 데이터들이 0(평균)을 중심으로 ±2 범위 내에 안정적으로 분포하며 고유의 패턴을 유지한다.")

"""
[최종 결론]
이상치가 산발적으로 발생하는 실제 제조/센서 데이터 환경에서는 
정규화(Min-Max)를 사용할 경우 모델이 정상 데이터의 미세한 변화를 학습하지 못할 위험이 크다.
반면 표준화는 이상치 충격을 분산시켜 데이터의 본질적인 특징을 훨씬 잘 보존한다.
"""
