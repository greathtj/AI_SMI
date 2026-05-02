import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. 가상의 제조 데이터 생성 (실제 현장 데이터라고 가정)
# 온도(temp), 압력(press), 진동(vib) -> 결과(target: 0은 정상, 1은 불량)
np.random.seed(42)
data_size = 1000

data = {
    'temp': np.random.uniform(60, 100, data_size),   # 60~100도
    'press': np.random.uniform(5, 15, data_size),    # 5~15 bar
    'vib': np.random.uniform(0.1, 0.5, data_size),   # 0.1~0.5 mm/s
}

df = pd.DataFrame(data)

# [불량 조건 설정]: 온도가 90도 이상이고 압력이 8bar 이하일 때 불량일 확률이 높다고 가정
df['target'] = np.where((df['temp'] > 90) & (df['press'] < 8), 1, 0)
# 약간의 노이즈 추가 (현실 데이터는 완벽하지 않으므로)
noise = np.random.choice([0, 1], size=data_size, p=[0.95, 0.05])
df['target'] = np.where(np.random.rand(data_size) < 0.05, noise, df['target'])

print("--- 데이터 샘플 (상위 5개) ---")
print(df.head())
print("\n")

# 2. 데이터 분리 (학습용과 테스트용)
X = df[['temp', 'press', 'vib']] # 입력 변수 (공정 데이터)
y = df['target']                # 정답 (불량 여부)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. AI 모델 생성 및 학습 (Random Forest 분류기 사용)
# 숲(Forest)을 만들어 여러 결정 나무가 함께 투표하여 결과를 내는 방식
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. 예측 및 평가
predictions = model.predict(X_test)
print(f"모델 예측 정확도: {accuracy_score(y_test, predictions):.2%}")
print("\n--- 상세 분석 보고서 ---")
print(classification_report(y_test, predictions))

# 5. [실전] 새로운 데이터가 들어왔을 때 불량 예측하기

new_product = pd.DataFrame([[95, 6, 0.4]], columns=['temp', 'press', 'vib']) 

prediction = model.predict(new_product)
probability = model.predict_proba(new_product)[0][1] # 불량일 확률

if prediction[0] == 1:
    print(f"\n⚠️ 경고: 이 제품은 불량일 가능성이 매우 높습니다! (확률: {probability:.2%})")
    print("즉시 공정 변수를 확인하고 조치하십시오.")
else:
    print(f"\n✅ 정상: 이 제품은 안전합니다. (불량 확률: {probability:.2%})")
