import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. 데이터 생성 (제조 현장 시나리오)
# 시나리오 A: [양의 상관관계] 기계 온도(X)와 내부 압력(y)의 관계
# 시나리오 B: [음의 상관관계] 작업자 숙련도(X)와 조립 시간(y)의 관계

# 데이터 준비
# 온도가 높아질수록 압력이 함께 상승하는 데이터를 생성한다.
temp = np.array([20, 22, 25, 28, 30, 32, 35, 38, 40, 42]).reshape(-1, 1)
pressure = 2.5 * temp + 15 + np.random.normal(0, 3, temp.shape) # y = 2.5x + 15 + 노이즈

# 숙련도가 높아질수록 작업 시간이 줄어드는 데이터를 생성한다.
experience = np.array([1, 2, 3, 5, 7, 10, 12, 15, 18, 20]).reshape(-1, 1)
task_time = -2 * experience + 50 + np.random.normal(0, 2, experience.shape) # y = -2x + 50 + 노이즈

# 2. 회귀 모델 학습 (선형 회귀)
# 데이터를 가장 잘 설명하는 직선(모델)을 찾아낸다.
model_a = LinearRegression().fit(temp, pressure)
model_b = LinearRegression().fit(experience, task_time)

# 회귀 직선을 그리기 위한 예측값을 계산한다.
pressure_pred = model_a.predict(temp)
time_pred = model_b.predict(experience)

# 3. 데이터 시각화 (영어 라벨 유지, 설명은 주석 참조)
# 폰트 에러를 방지하기 위해 기본 폰트를 사용한다.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 왼쪽 그래프: 양의 상관관계 (온도와 압력)
ax1.scatter(temp, pressure, color='blue', label='Actual Data')
ax1.plot(temp, pressure_pred, color='red', linewidth=2, label='Regression Line (Pred)')
corr_a = pd.Series(temp.flatten()).corr(pd.Series(pressure.flatten()))
ax1.set_title(f'Positive Correlation: Temp vs Pressure\n(Correlation: {corr_a:.2f})')
ax1.set_xlabel('Temperature (C)')
ax1.set_ylabel('Internal Pressure (bar)')
ax1.legend()
ax1.grid(True, linestyle='--')

# 오른쪽 그래프: 음의 상관관계 (숙련도와 시간)
ax2.scatter(experience, task_time, color='green', label='Actual Data')
ax2.plot(experience, time_pred, color='orange', linewidth=2, label='Regression Line (Pred)')
corr_b = pd.Series(experience.flatten()).corr(pd.Series(task_time.flatten()))
ax2.set_title(f'Negative Correlation: Experience vs Task Time\n(Correlation: {corr_b:.2f})')
ax2.set_xlabel('Experience (Years)')
ax2.set_ylabel('Assembly Time (Min)')
ax2.legend()
ax2.grid(True, linestyle='--')

plt.tight_layout()
plt.show()

# 4. 분석 결과 요약 출력
print("--- [분석 결과 보고서] ---")
print(f"A 공정(온도): 온도가 1도 올라갈 때마다, 압력은 약 {model_a.coef_[0][0]:.2f} bar만큼 상승한다.")
print(f"B 공정(숙련도): 숙련도가 1년 쌓일 때마다, 작업 시간은 약 {abs(model_b.coef_[0][0]):.2f}분만큼 단축된다.")
print("\n[팁] 수집한 센서 데이터만 이 모델에 넣으면")
print("기계의 다음 행동을 예측하는 '디지털 점쟁이'가 될 수 있다!")
