import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. 데이터 준비 (지난 12개월간의 실제 생산량)
# X: 월(1월~12월), y: 생산량(단위: 개)
months = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(-1, 1)
production = np.array([800, 820, 850, 840, 880, 910, 900, 930, 960, 950, 980, 1010])

# 2. AI 모델 생성 및 학습
model = LinearRegression()
model.fit(months, production)

# 3. 다음 달(13월) 생산량 예측
next_month = np.array([[13]])
prediction = model.predict(next_month)

print(f"내년 1월(13월차) 예상 생산량: {prediction[0]:.2f}개")

# 4. 시각화 (추세선 확인)
plt.scatter(months, production, color='blue', label='Actual')
plt.plot(months, model.predict(months), color='red', label='Trend Line')
plt.title("Monthly Production Trend")
plt.xlabel("Month")
plt.ylabel("Quantity")
plt.legend()
plt.show()
