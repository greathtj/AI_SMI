import torch
import torch.nn as np_torch # torch.nn의 줄임말
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 1. 가상 데이터 생성 (TensorFlow 예제와 동일한 조건)
np.random.seed(42)
time = np.arange(365)
production = 500 + (time * 0.5) + (np.sin(time * 2 * np.pi / 30) * 50) + np.random.randn(365) * 10
data = production.reshape(-1, 1).astype(np.float32)

# 2. 데이터 정규화 (0~1 사이 값으로 변환)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 3. 시퀀스 데이터 생성 함수
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

WINDOW_SIZE = 30
X, y = create_sequences(scaled_data, WINDOW_SIZE)

# PyTorch 전용 텐서(Tensor) 타입으로 변환
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# 학습/테스트 데이터 분리 (80% 학습, 20% 테스트)
train_size = int(len(X) * 0.8)
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

# 4. LSTM 모델 클래스 정의
class ProductionLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50):
        super(ProductionLSTM, self).__init__()
        self.hidden_size = hidden_size
        # LSTM 층: 입력값 1개 -> 은닉 상태 50개
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # 최종 출력 층: 은닉 상태 50개 -> 예측값 1개
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # lstm_out: 모든 시점의 출력값, (hn, cn): 마지막 시점의 은닉 상태
        lstm_out, (hn, cn) = self.lstm(x)
        # 마지막 시점의 출력값만 사용하여 최종 예측값 도출
        out = self.linear(lstm_out[:, -1, :])
        return out

# 모델, 손실함수, 옵티마이저 설정
model = ProductionLSTM()
criterion = nn.MSELoss() # 평균제곱오차
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 5. 모델 학습 (Training Loop)
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad() # 기울기 초기화
    
    outputs = model(X_train)
    loss = criterion(outputs, y_train) # 오차 계산
    
    loss.backward() # 역전파 (오차를 뒤로 전달)
    optimizer.step() # 가중치 업데이트
    
    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 6. 예측 및 시각화
model.eval()
with torch.no_grad():
    predictions = model(X_test).numpy()
    actuals = y_test.numpy()

# 정규화 해제 (원래 수치로 복원)
predictions_rescaled = scaler.inverse_transform(predictions)
actuals_rescaled = scaler.inverse_transform(actuals)

plt.figure(figsize=(12, 6))
plt.plot(actuals_rescaled, label='Actual Production', color='blue')
plt.plot(predictions_rescaled, label='PyTorch LSTM Prediction', color='red', linestyle='--')
plt.title("Production Volume Prediction using PyTorch LSTM")
plt.legend()
plt.show()

# 7. 내일의 생산량 예측
last_30_days = torch.FloatTensor(scaled_data[-WINDOW_SIZE:]).view(1, WINDOW_SIZE, 1)
with torch.no_grad():
    tomorrow_pred_scaled = model(last_30_days)
    tomorrow_pred = scaler.inverse_transform(tomorrow_pred_scaled.numpy())

print(f"\n🚀 AI 예측 결과: 내일의 예상 생산량은 {tomorrow_pred[0][0]:.2f}개입니다.")
