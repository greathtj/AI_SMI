import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# 1. 가상 데이터 생성 (제조 현장 시나리오)
# 정상 데이터: 온도(200~210도), 압력(40~50bar) 근처에 100개 생성
np.random.seed(42)
normal_data = np.random.normal(loc=[205, 45], scale=[2, 2], size=(100, 2))

# 이상치 데이터: 일부러 범위를 벗어난 데이터 5개 추가
# (예: 온도는 정상인데 압력이 너무 낮거나, 온도가 너무 높은 경우 등)
outliers = np.array([
    [205, 30],  # 온도 정상, 압력 매우 낮음
    [225, 45],  # 온도 매우 높음, 압력 정상
    [190, 55],  # 온도 낮고, 압력 높음
    [210, 60],  # 온도 정상, 압력 매우 높음
    [215, 35]   # 복합 이상
])

# 전체 데이터 합치기
X = np.vstack([normal_data, outliers])
df = pd.DataFrame(X, columns=['Temperature', 'Pressure'])

# 2. Isolation Forest 모델 설정 및 학습
# contamination: 전체 데이터 중 이상치 비율 (여기서는 약 5%로 설정)
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(df)

# 3. 이상치 판별
# predict() 결과: 1은 정상(Inlier), -1은 이상치(Outlier)
df['anomaly'] = model.predict(df)
df['score'] = model.decision_function(df.drop('anomaly', axis=1)) # 이상치 점수 (낮을수록 더 이상함)

# 4. 결과 출력
print("--- 이상치 검출 결과 (상위 5개) ---")
print(df[df['anomaly'] == -1])

# 5. 시각화
plt.figure(figsize=(10, 6))
# 정상 데이터는 파란색, 이상치는 빨간색으로 표시
plt.scatter(df.loc[df['anomaly'] == 1, 'Temperature'], 
            df.loc[df['anomaly'] == 1, 'Pressure'], 
            c='blue', label='Normal', alpha=0.6)
plt.scatter(df.loc[df['anomaly'] == -1, 'Temperature'], 
            df.loc[df['anomaly'] == -1, 'Pressure'], 
            c='red', label='Outlier', edgecolors='k', s=100)

plt.title('Isolation Forest Outlier Detection (Temp vs Pressure)')
plt.xlabel('Temperature (°C)')
plt.ylabel('Pressure (bar)')
plt.legend()
plt.grid(True, linestyle='--')
plt.show()
