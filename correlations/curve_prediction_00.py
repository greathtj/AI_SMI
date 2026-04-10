import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# 1. 데이터 생성: 직선 관계 vs 곡선 관계
# - 선형(Linear): 온도가 오르면 정직하게 압력이 오른다.
# - 비선형(Non-linear): 전력이 오르면 생산성이 오르다가, 과열 시 급격히 떨어진다 (산 모양).

rng = np.random.RandomState(42)
n_samples = 100

# 선형 데이터
temp = rng.uniform(20, 50, n_samples)
pressure = 2 * temp + rng.normal(0, 2, n_samples)

# 비선형 데이터 (산 모양 곡선)
power = rng.uniform(0, 100, n_samples)
# 최적 전력 50에서 최대 생산성, 그 이상은 과열로 하락
productivity = 100 - 0.1 * (power - 50)**2 + rng.normal(0, 5, n_samples)

df = pd.DataFrame({
    'Temperature': temp,
    'Pressure': pressure,
    'Power': power,
    'Productivity': productivity
})

# 2. 모델 학습 및 곡선 예측 데이터 준비
# 시각화를 위해 촘촘한 가상의 입력값(X_test)을 만든다.
temp_range = np.linspace(20, 50, 100).reshape(-1, 1)
power_range = np.linspace(0, 100, 100).reshape(-1, 1)

# [왼쪽 그래프용] 선형 관계 모델 학습
model_lin_left = LinearRegression().fit(df[['Temperature']], df['Pressure'])
line_pred_left = model_lin_left.predict(temp_range)

# [오른쪽 그래프용] 두 가지 모델 비교 학습
# 1. 전통적인 직선 모델 (Linear Regression)
model_lin_right = LinearRegression().fit(df[['Power']], df['Productivity'])
line_pred_right = model_lin_right.predict(power_range)

# 2. 유연한 AI 모델 (Random Forest)
model_nonlin_right = RandomForestRegressor(n_estimators=100, random_state=42).fit(df[['Power']], df['Productivity'])
curve_pred_right = model_nonlin_right.predict(power_range)

# 3. 시각화 비교
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 왼쪽: 직선 관계 (전통 통계가 잘 맞음)
sns.scatterplot(x='Temperature', y='Pressure', data=df, ax=ax1, color='blue', alpha=0.6)
ax1.plot(temp_range, line_pred_left, color='red', linewidth=3, label='AI Linear Prediction')
ax1.set_title("Linear Relationship: Temperature vs Pressure\n(Easy for Traditional Stats)")
ax1.legend()

# 오른쪽: 곡선 관계 (직선 모델 vs 유연한 모델 비교)
sns.scatterplot(x='Power', y='Productivity', data=df, ax=ax2, color='green', alpha=0.6)
# 전통적인 직선 모델은 데이터의 흐름을 무시하고 중간을 가로지른다.
ax2.plot(power_range, line_pred_right, color='gray', linestyle=':', linewidth=2, label='Traditional Linear (Fail)')
# 유연한 모델은 데이터의 꺾임(산 모양)을 그대로 따라간다.
ax2.plot(power_range, curve_pred_right, color='red', linestyle='--', linewidth=3, label='AI Flexible Curve (Success)')
ax2.set_title("Non-linear Relationship: Power vs Productivity\n(Linear vs Flexible Model)")
ax2.legend()

plt.tight_layout()
plt.show()

# 4. 모델 성능 비교 (수사 보고서)
def analyze_relation(X_col, y_col):
    X = df[[X_col]]
    y = df[y_col]
    
    # 직선 모델 (선형 회귀)
    lr = LinearRegression().fit(X, y)
    lr_pred = lr.predict(X)
    lr_score = r2_score(y, lr_pred)
    
    # 유연한 모델 (랜덤 포레스트 - 딥러닝과 유사한 유연함)
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    rf_pred = rf.predict(X)
    rf_score = r2_score(y, rf_pred)
    
    return lr_score, rf_score

lin_lr, lin_rf = analyze_relation('Temperature', 'Pressure')
non_lr, non_rf = analyze_relation('Power', 'Productivity')

print("--- [데이터 수사 보고서: 직선 vs 곡선] ---")
print(f"1. [온도-압력] 직선 관계 분석")
print(f"   - 직선 모델 정확도: {lin_lr*100:.1f}% (성공)")
print(f"   - 유연한 모델 정확도: {lin_rf*100:.1f}% (성공)")

print(f"\n2. [전력-생산성] 곡선 관계 분석")
print(f"   - 직선 모델 정확도: {non_lr*100:.1f}% (실패! 패턴을 전혀 읽지 못함)")
print(f"   - 유연한 모델 정확도: {non_rf*100:.1f}% (성공! 산 모양 패턴 검거)")

print("\n[결론] 오른쪽 그래프의 회색 점선(직선 모델)을 보면 데이터의 산 모양을 무시한 채")
print("중간을 멍하니 지나가고 있다. 반면 빨간 점선(유연한 모델)은 데이터의 흐름을 정확히 따라간다.")
