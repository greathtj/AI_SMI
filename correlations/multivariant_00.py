import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 제조 현장 가상 데이터 생성 (다변량 시나리오)
# - Temperature: 가열 온도
# - Pressure: 내부 압력 (온도와 강한 양의 상관관계)
# - Vibration: 설비 진동 (온도가 너무 높을 때 불규칙하게 상승)
# - Quality_Score: 최종 품질 점수 (압력과 진동에 의해 결정됨)

np.random.seed(42)
n_samples = 100

temp = np.random.normal(30, 5, n_samples)
# 압력은 온도에 2.5배 비례하고 약간의 노이즈가 섞인다 (양의 상관관계).
pressure = 2.5 * temp + np.random.normal(0, 2, n_samples)
# 진동은 온도와 약한 상관관계가 있으나 대체로 독립적이다.
vibration = 0.5 * temp + np.random.normal(10, 5, n_samples)
# 품질 점수는 압력이 적당하고 진동이 낮을수록 높다 (복합 관계).
quality_score = 100 - (abs(pressure - 75) + vibration) + np.random.normal(0, 2, n_samples)

# 데이터프레임으로 변환한다.
df = pd.DataFrame({
    'Temperature': temp,
    'Pressure': pressure,
    'Vibration': vibration,
    'Quality': quality_score
})

# 2. 다변량 상관분석 시각화 (Pair Plot)
# 모든 변수 간의 관계를 바둑판 형태로 한꺼번에 그린다.
# 대각선은 각 데이터의 분포를, 나머지는 두 데이터 간의 산점도를 보여준다.
sns.set_theme(style="whitegrid")
g = sns.pairplot(df, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k'})
g.fig.suptitle("Multivariate Analysis: Factory Sensor Map", y=1.02)

plt.show()

# 3. 상관계수 행렬(Correlation Matrix) 출력
# 숫자로 된 '궁합 지도'를 출력하여 정량적으로 확인한다.
print("--- [센서 간 상관계수 행렬] ---")
print(df.corr().round(2))

# 4. 분석 결과 해석
print("\n--- [데이터 분석 결과 보고서] ---")
print("1. 온도(Temperature)와 압력(Pressure)은 1.0에 가까운 아주 강력한 양의 상관관계를 보인다.")
print("2. 품질(Quality)은 압력 및 진동과 복합적으로 연결되어 있어 단순한 직선보다 복잡한 분포를 가진다.")
print("3. 이처럼 얽혀 있는 관계는 딥러닝 모델에 넣었을 때 비로소 '진짜 불량 원인'을 찾아낼 수 있다.")

print("\n[팁] 이 도구를 활용해 수십 개의 센서 중 '함께 움직이는 그룹'을 찾아내는 것만으로도")
print("분석의 절반은 성공이다!")
