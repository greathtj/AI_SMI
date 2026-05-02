import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 1. 가상 데이터 생성 (변수가 많을수록 현실적임)
np.random.seed(42)
data_size = 500

data = {
    'uptime': np.random.uniform(70, 98, data_size),    # 설비 가동률 (%)
    'workers': np.random.randint(5, 15, data_size),    # 투입 작업자 수 (명)
    'orders': np.random.randint(1000, 2000, data_size), # 주문량 (개)
}
df = pd.DataFrame(data)

# [생산량 결정 규칙]: 가동률과 작업자 수가 많을수록 생산량이 늘어남
df['actual_prod'] = (df['uptime'] * 5) + (df['workers'] * 10) + (np.random.randn(data_size) * 10)

# 2. 데이터 분리
X = df[['uptime', 'workers', 'orders']]
y = df['actual_prod']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. AI 모델 학습 (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. 예측 및 오차 확인
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print(f"평균 예측 오차(MAE): {mae:.2f}개")

# 5. [실전 적용] 내일의 조건으로 생산량 예측하기
# 조건: 가동률 92%, 작업자 10명, 주문량 1500개
tomorrow_condition = pd.DataFrame([[92, 10, 1500]], columns=['uptime', 'workers', 'orders'])
tomorrow_pred = model.predict(tomorrow_condition)

print(f"\n내일 예상 생산량: {tomorrow_pred[0]:.2f}개")
