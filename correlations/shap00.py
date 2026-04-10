import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor

# 1. 제조 현장 가상 데이터 생성
# 경고 해결: np.random.seed 대신 전용 RNG 객체를 사용하여 일관성을 유지한다.
rng = np.random.RandomState(42)
n_samples = 200

temp = rng.normal(30, 5, n_samples)
pressure = 2.5 * temp + rng.normal(0, 2, n_samples)
vibration = rng.normal(10, 3, n_samples)
# 품질 점수는 압력이 적당(75 근처)하고 진동이 낮을수록 높다.
quality_score = 100 - (abs(pressure - 75) + 2 * vibration) + rng.normal(0, 1, n_samples)

df = pd.DataFrame({
    'Temperature': temp,
    'Pressure': pressure,
    'Vibration': vibration,
    'Quality': quality_score
})

# 2. AI 모델 학습
X = df[['Temperature', 'Pressure', 'Vibration']]
y = df['Quality']
model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)

# 3. SHAP 수사관 투입 (설명 가능한 AI 분석)
# 모델의 예측 원인을 수학적으로 분해하여 각 변수의 기여도를 계산한다.
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 4. 시각화 (SHAP Summary Plot)
# 각 점은 하나의 데이터(품질 판정 건수)를 의미한다. 
# 빨간색일수록 해당 센서 값이 높음을, 오른쪽으로 치우칠수록 품질 점수를 높였음을 뜻한다.
plt.figure(figsize=(10, 6))

# 경고 해결: shap 라이브러리 내부에서 전역 난수(np.random.seed)를 호출하여 발생하는 경고를 방지하기 위해 
# 시각화 시 데이터의 정렬을 고정하거나 환경을 안정화한다.
# 최신 shap 버전에서는 시각화 시 발생하는 이 경고가 라이브러리 자체의 이슈인 경우가 많으므로, 
# 코드 상에서 가능한 한 명시적으로 데이터를 전달한다.
shap.summary_plot(shap_values, X, show=False, plot_type="dot")

plt.title("SHAP Analysis: Why did AI make this decision?")
plt.show()

# 5. 분석 결과 해석 (평서체)
print("--- [SHAP 정밀 수사 보고서] ---")
print("1. SHAP 분석은 단순히 '중요하다'를 넘어, 해당 변수가 점수를 '올렸는지' '깎았는지'를 보여준다.")
print("2. 그래프에서 Vibration(진동)이 높을수록(빨간색), SHAP 수치가 왼쪽(음수)으로 이동한다.")
print("   - 즉, 진동 상승은 품질 점수를 깎아먹는 '확실한 주범'임을 알 수 있다.")
print("3. Pressure(압력)의 경우, 특정 범위를 벗어나면 품질에 부정적인 영향을 주는 복잡한 양상을 띤다.")

print("\n[결론] SHAP을 사용하면 '오늘 오전 10시 불량은 진동 때문인가, 온도 때문인가?'라는 질문에")
print("개별 데이터 단위로 명확한 근거를 제시할 수 있다.")

print("\n[팁] 전문적인 프로그래밍 스킬이 없더라도, 센서 데이터만 잘 모아둔다면")
print("이러한 AI 도구들이 수사관처럼 공정의 비밀을 하나씩 밝혀준다!")
