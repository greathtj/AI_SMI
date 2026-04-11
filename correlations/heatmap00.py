import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 가상 공장 데이터 생성 (히트맵용)
# 온도, 압력, 진동, 습도, 불량률 사이의 관계 설정
# 실무에서 흔히 발생하는 상관관계를 재현합니다.
np.random.seed(42)
n_samples = 100

temperature = np.random.randn(n_samples)
# 압력은 온도와 강한 양의 상관관계를 가짐 (온도가 오르면 압력도 오름)
pressure = 0.8 * temperature + np.random.normal(0, 0.5, n_samples)
# 습도는 불량률과 어느 정도 상관관계를 가짐
humidity = np.random.randn(n_samples)
defect_rate = 0.6 * humidity + np.random.normal(0, 0.7, n_samples)
# 진동은 독립적인 노이즈로 설정
vibration = np.random.randn(n_samples)

data = {
    'Temperature': temperature,
    'Pressure': pressure,
    'Vibration': vibration,
    'Humidity': humidity,
    'Defect_Rate': defect_rate
}
df = pd.DataFrame(data)

# 상관관계 계산 (히트맵의 기초: -1부터 1 사이의 값)
corr = df.corr()

# 2. 시각화: 공장 데이터 인맥 지도 (히트맵)
plt.figure(figsize=(10, 8))

# sns.heatmap 설정 설명:
# annot=True: 칸 안에 숫자를 표시
# cmap='RdBu_r': 빨간색(양의 상관)과 파란색(음의 상관)으로 표시
# center=0: 0을 기준으로 색상을 배분
sns.heatmap(corr, annot=True, fmt=".2f", cmap='RdBu_r', center=0, linewidths=.5)

plt.title("Factory Data Heatmap: Relationship Map\n(Who is collaborating with whom?)", fontsize=16)
plt.show()

# 3. 분석 결과 해석 (평서체)
print("--- [데이터 수사 보고서: 공장 인맥 지도] ---")
print("1. 히트맵의 빨간색 칸을 주목하라.")
print("   - Temperature(온도)와 Pressure(압력)의 교차점이 0.81로 매우 빨갛다.")
print("   - 이는 '온도가 오르면 압력도 확실히 오른다'는 강력한 우정(상관관계)을 뜻한다.")
print("\n2. Humidity(습도)와 Defect_Rate(불량률)의 관계도 확인하라.")
print("   - 상관계수가 0.5를 넘어가고 있다. 습도가 높아질 때 불량이 발생할 확률이 높다는 증거다.")
print("\n3. Vibration(진동)은 다른 센서들과 색깔이 연하다.")
print("   - 이 녀석은 현재 다른 데이터들과 따로 노는 '독고다이' 상태임을 알 수 있다.")

print("\n[결론] 히트맵은 수십 개의 센서 중 누가 한패가 되어 움직이는지 보여주는 '종합 상황판'이다.")
